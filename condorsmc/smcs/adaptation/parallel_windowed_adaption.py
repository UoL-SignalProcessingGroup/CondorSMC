import autograd.numpy as np

from scipy.stats import multivariate_normal
from tqdm import tqdm
import warnings

from ..executor.sequential import SequentialExecutor
from ..weight_updater import WeightUpdater
from ..resampling.multinomial import MultinomialResampling
from ..importance_sampling import estimate_moments
from ..recycling import NoRecycling
from ..state import SMCState


class StepSizeAdaption:
    def __init__(self, target, forward_kernel, lkernel, executor=SequentialExecutor(), rng=np.random.default_rng()):
        self.target = target
        self.forward_kernel = forward_kernel
        self.lkernel = lkernel
        self.executor = executor
        self.rng = rng

        # Adaption parameters
        self.t0 = 10
        self.gamma = 0.05
        self.delta = 0.8

    def find_reasonable_epsilon(self, smc_state):
        step_size = np.ones(smc_state.num_particles_local)

        for i in range(smc_state.num_particles_local):
            epsilon = 1

            delta_H = self.CheckEpsilon(smc_state.positions[i], smc_state.log_pdf_grads[i], epsilon)

            direction = np.where(delta_H > np.log(0.5), 1, -1)

            while(True):
                
                delta_H = self.CheckEpsilon(smc_state.positions[i], smc_state.log_pdf_grads[i], epsilon)
    
                if(np.isnan(delta_H)):
                    direction = -1
                    delta_H = -np.inf

                if np.all(direction == 1) and not np.any(delta_H > np.log(0.5)):
                    break
                elif np.all(direction == -1) and not np.any(delta_H < np.log(0.5)):
                    break
                else:
                    epsilon = np.where(direction == 1, 2.0 * epsilon, 0.5 * epsilon)
                
            step_size[i] = epsilon

        self.forward_kernel.set_step_size(step_size)

    def Leapfrog(self, x, r, grad_x, step_size):
    
        """
        Description
        -----------
        Performs a single Leapfrog step returning the final position, momentum and gradient.
        """
        r = np.add(r, (step_size/2)*grad_x)
        x = np.add(x, step_size*r)
        grad_x = self.target.logpdfgrad(x, phi=1.0)
  
        r = np.add(r, (step_size/2)*grad_x)
        
        return x, r, grad_x

    def CheckEpsilon(self, x, grad_x, epsilon):
        # samples a new momentum
        r = self.forward_kernel.momentum_proposal.rvs(1, random_state=self.rng)

        #Take a leapfrog step with momentum r
        # x_prime, r_prime, _ = self.forward_kernel.NUTSLeapfrog(x, r, grad_x, 1.0, epsilon, print_text=False)
        x_prime, r_prime, _ = self.Leapfrog(x, r, grad_x, epsilon)

        # Calculate the (log) Hamiltonians at the end and start of the proposal
        H0 = self.target.logpdf(x, phi=1.0) - (0.5 * np.dot(r, r))
        H1 = self.target.logpdf(x_prime, phi=1.0) - (0.5 * np.dot(r_prime, r_prime))

        #calculate the difference in the Hamiltonian
        delta_H = H1 - H0

        return delta_H

    def adapt(self, acceptance, Hbar, mu, k):
        eta = 1. / float(k + self.t0)
        Hbar = (1. - eta) * Hbar + eta * (self.delta - acceptance)
    
        self.forward_kernel.step_size = np.exp(mu - np.sqrt(k) / self.gamma * Hbar)

        return Hbar

    def resample_epsilon(self, smc_state,  Hbar, mu):
        i = np.linspace(0, smc_state.num_particles-1, smc_state.num_particles, dtype=int)
        i_new = self.rng.choice(i, smc_state.num_particles, p=smc_state.normalised_weights)
        self.forward_kernel.step_size = self.forward_kernel.step_size[i_new]
        smc_state.positions = smc_state.positions[i_new]
        smc_state.momenta = smc_state.momenta[i_new]
        Hbar = Hbar[i_new]
        mu = mu[i_new]

        smc_state.log_pdfs = smc_state.target.logpdf(smc_state.positions)
        smc_state.log_pdf_grads = smc_state.target.logpdfgrad(smc_state.positions)
        smc_state.log_weights = np.full(smc_state.num_particles_local, smc_state.log_normalising_constant - np.log(smc_state.num_particles))
        smc_state.normalised_weights = np.exp(smc_state.log_weights)

        return smc_state, Hbar, mu


class MassMatrixAdaption:
    def __init__(self, target, forward_kernel, lkernel, executor=SequentialExecutor(), rng=np.random.default_rng()):
        self.target = target
        self.forward_kernel = forward_kernel
        self.lkernel = lkernel
        self.executor = executor
        self.rng = rng

    def adapt(self, smc_state):
        mean = self.executor.weighted_sum(smc_state.positions, smc_state.normalised_weights)
        var = self.executor.weighted_sum(np.square(smc_state.positions - mean), smc_state.normalised_weights)
        var = var[:smc_state.target.dim]

        metric = 1 / var
        
        self.forward_kernel.set_inv_metric(var)
        self.forward_kernel.momentum_proposal = multivariate_normal(mean=np.zeros(self.target.dim), cov=np.diag(metric))


class ParallelWindowedAdaption:
    def __init__(self, target, forward_kernel, lkernel, recycling=NoRecycling(), executor=SequentialExecutor(), rng=np.random.default_rng()):
        self.target = target
        self.forward_kernel = forward_kernel
        self.lkernel = lkernel
        self.recycling = recycling
        self.executor = executor
        self.rng = rng

        # Initialize the step size adaption and mass matrix adaption objects
        self.step_size_adaption = StepSizeAdaption(target, forward_kernel, lkernel, executor, rng)
        self.mass_matrix_adaption = MassMatrixAdaption(target, forward_kernel, lkernel, executor, rng)

        # Sampler components
        self.resample_method = MultinomialResampling(executor=self.executor, rng=self.rng)
        self.weight_updater = WeightUpdater(self.target, self.forward_kernel, self.lkernel, self.executor)

    def adapt(self, smc_state, adaption_statistics):
        # Find reasonable initial step size
        self.step_size_adaption.find_reasonable_epsilon(smc_state)

        # Set adaption parameters
        Hbar = np.zeros(smc_state.num_particles_local)
        count = 0
        self.mu = np.log(10 * self.forward_kernel.step_size)

        # Main adaption steps
        count, Hbar, smc_state = self.step(smc_state, adaption_statistics, count, Hbar, 0.5, 5, 30)
        self.mass_matrix_adaption.adapt(smc_state)

        count, Hbar, smc_state = self.step(smc_state, adaption_statistics, count, Hbar, 0.4, 3, 40)
        self.mass_matrix_adaption.adapt(smc_state)

        count, Hbar, smc_state = self.step(smc_state, adaption_statistics, count, Hbar, 0.2, 3, 40)
        self.mass_matrix_adaption.adapt(smc_state)

        count, Hbar, smc_state = self.step(smc_state, adaption_statistics, count, Hbar, 0.1, 3, 50)

        # Set final step size
        final_step_size = np.ones(smc_state.num_particles_local) * self.executor.weighted_sum(self.forward_kernel.step_size, smc_state.normalised_weights)
        self.forward_kernel.set_step_size(final_step_size)

        return smc_state, self.forward_kernel, self.lkernel

    def step(self, smc_state, adaption_statistics, count, Hbar, Tol, WindowSize, MaxWindowRange):
        Step_converged = False
        Array_of_steps = []

        while_count = 1
        while(Step_converged == False and while_count < MaxWindowRange):
            # Move samples
            smc_state = self.forward_kernel.propose(smc_state)

            # Adapt samples
            Hbar = self.step_size_adaption.adapt(smc_state.acceptance_rate, Hbar, self.mu, count)

            # Update weights
            smc_state = self.weight_updater.update(smc_state)

            # Calculate recycling constant
            smc_state = self.recycling.calculate_constant(smc_state)

            # Estimate moments and diagnostics
            adaption_statistics.update(smc_state)

            # Resample if needed
            ess = smc_state.ess
            if ess < 0.5 * smc_state.num_particles:
                smc_state, Hbar, self.mu = self.step_size_adaption.resample_epsilon(smc_state, Hbar, self.mu)

            Array_of_steps.append(self.executor.weighted_sum(self.forward_kernel.step_size, smc_state.normalised_weights))

            if while_count > WindowSize-1:
                if(np.abs(np.std(Array_of_steps[-WindowSize:])/np.mean(Array_of_steps[-WindowSize:])) < Tol and ess > smc_state.num_particles / 2):
                    Step_converged = True

            while_count += 1
            count += 1

        return count, Hbar, smc_state
