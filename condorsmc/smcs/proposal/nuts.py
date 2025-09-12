import autograd.numpy as np

from .base import ProposalBase
from .utils import hmc_accept_reject

MAX_TREE_DEPTH = 10


"""
THIS CLASS IS A MERGE OF THE HMCPROPOSAL CLASS AND THE NUTS PROPSAL FROM
THE BIG HYPOTHESES PYTHON REPOSITORY. IT IS NOT YET FINISHED AND NEEDS TIDYING
UP AND TESTING.

CURRENTLY RUNS SEQUENTIALLY AND IS NOT VECTORISED.
"""

class NUTSProposal(ProposalBase):
    """Hamiltonian Monte Carlo Proposal

    Propagate samples using a Hamiltonian Monte Carlo (HMC) proposal. HMC propagates
    samples using an approximate simulation of the Hamiltonian dynamics of a system.

    [1] https://mc-stan.org/docs/2_19/reference-manual/hamiltonian-monte-carlo.html

    Attributes:
        dim: Dimensionality of the system.
        target: Target distribution of interest.
        step_size: Step size for the leapfrog integrator.
        num_steps: Number of leapfrog steps to take.
        momentum_proposal: Momentum proposal distribution.
        random_num_steps: Whether or not to use a random number of leapfrog steps.
    """

    def __init__(
        self,
        target,
        momentum_proposal,
        step_size: float = 1.0,
        rng = np.random.default_rng(),
    ):
        self.target = target
        self.momentum_proposal = momentum_proposal
        self.inv_metric = np.ones(target.dim)
        self.step_size = step_size
        self.rng = rng

    def propose(self, smc_state):
        x_prime = np.zeros((smc_state.num_particles_local, self.target.dim))
        smc_state.momenta = self.momentum_proposal.rvs(smc_state.num_particles_local, random_state=self.rng)
        if self.target.dim == 1:
            smc_state.momenta = smc_state.momenta[:, np.newaxis]
        acceptance_rate = np.zeros(smc_state.num_particles_local)
        depths = np.zeros(smc_state.num_particles_local)
        r_prime = np.zeros((smc_state.num_particles_local, self.target.dim))

        # Temporary fix for tempering, update logpdfgrad based on latest temperature
        smc_state.log_pdf_grads = self.target.logpdfgrad(smc_state.positions, phi=smc_state.phi)

        for i in range(smc_state.num_particles_local):
            step_size = self.step_size if isinstance(self.step_size, float) else self.step_size[i]
            x_prime[i], r_prime[i], acceptance_rate[i], depths[i] = self.generate_nuts_samples(
                smc_state.positions[i], smc_state.momenta[i], smc_state.log_pdf_grads[i], step_size, phi=smc_state.phi
            )

        smc_state.update_samples(x_prime, r_prime)
        smc_state.acceptance_rate = acceptance_rate

        return smc_state

    def generate_nuts_samples(self, x0, r0, grad_x, step_size, phi: float = 1.0):

        """
        Description
        -----------
        Generates samples using the NUTS proposal, Based off Alg. 3 in [1]
        """
        
        logp = self.target.logpdf(x0, phi=phi)    
        #self.H0 = logp - 0.5 * np.dot(r0, r0.T)

        self.H0 = logp - 0.5 * np.dot(r0, np.multiply(self.inv_metric,r0).T)

        logu = float(self.H0 - self.rng.exponential(1))
        
        # initialize the NUTS tree 
        x = x0
        xminus = x0
        xplus = x0
        rminus = r0
        rplus = r0
        r = r0
        gradminus = grad_x
        gradplus = grad_x
 

        depth = 0  
        n = 1  
        stop = 0  

        while (stop == 0):
            # Using a Bernoulli trial choose a direction. -1 (backwards) or +1 (forwards)
            direction = int(2 * (self.rng.uniform(0,1) < 0.5) - 1)

            if (direction == -1):
                xminus, rminus, gradminus, _, _, _, xprime, rprime, nprime, stopprime, alpha, nalpha= self.build_tree(xminus, rminus, gradminus, logu, direction, depth,  step_size, phi)
            else:
                _, _, _, xplus, rplus, gradplus, xprime, rprime, nprime, stopprime, alpha, nalpha  = self.build_tree(xplus, rplus, gradplus, logu, direction, depth, step_size, phi)


            if (stopprime == 0 and self.rng.uniform() < min(1., float(nprime) / float(n))):
                x = xprime
                r = rprime

            n += nprime

            stop = stopprime or self.stop_criterion(xminus, xplus, rminus, rplus)           
            
            depth += 1
            
            if(depth > MAX_TREE_DEPTH):
                break
        
        acceptance = alpha/nalpha
        #print(depth-1)
        return x, r, acceptance, depth

    def build_tree(self, x, r, grad_x, logu, direction, depth, step_size, temperature=1.0):
        """
        Description
        -----------
        Generates samples using the recursive NUTS tree-building procedure [1]
        """
        if (depth == 0):
            xprime, rprime, gradprime = self.NUTSLeapfrog(x, r, grad_x, direction, step_size, temperature)
            logpprime = self.target.logpdf(xprime, phi=temperature)
            #joint = logpprime - 0.5 * np.dot(rprime, rprime.T)
            joint = logpprime - 0.5 * np.dot(rprime, np.multiply(self.inv_metric,rprime).T)
            
            nprime = int(logu < joint)
            stopprime = int((logu - 100.) >= joint)
            xminus = xprime
            xplus = xprime
            rminus = rprime
            rplus = rprime
            gradminus = gradprime
            gradplus = gradprime

            alphaprime = float(np.exp(joint-self.H0))
       
            if np.isnan(alphaprime):
                alphaprime = 0.0
            else:

                alphaprime = np.min([1.0, alphaprime])
                
            nalphaprime = 1

        else:
                                                                                         
            xminus, rminus, gradminus, xplus, rplus, gradplus, xprime, rprime,  nprime, stopprime, alphaprime, nalphaprime = self.build_tree(x, r, grad_x, logu, direction, depth - 1,  step_size, temperature)
            
            if (stopprime == 0):
                if (direction == -1):
                    xminus, rminus, gradminus, _, _, _, xprime2, rprime2, nprime2, stopprime2, alphaprime2, nalphaprime2  = self.build_tree(xminus, rminus, gradminus, logu, direction, depth - 1,  step_size, temperature)
                else:
                    _, _, _, xplus, rplus, gradplus, xprime2, rprime2, nprime2, stopprime2, alphaprime2, nalphaprime2   = self.build_tree(xplus, rplus, gradplus, logu, direction, depth - 1, step_size, temperature)           
               
                if (self.rng.uniform() < (float(nprime2) / max(float(int(nprime) + int(nprime2)), 1.))):
                    xprime = xprime2
                    rprime = rprime2

                nprime = int(nprime) + int(nprime2)
                stopprime = int(stopprime or stopprime2 or self.stop_criterion(xminus, xplus, rminus, rplus))
                alphaprime = alphaprime + alphaprime2
                nalphaprime = nalphaprime + nalphaprime2


        return xminus, rminus, gradminus, xplus, rplus, gradplus, xprime, rprime,  nprime, stopprime, alphaprime, nalphaprime

    def stop_criterion(self, xminus, xplus, rminus, rplus):
        """
        Description
        -----------
        Checks if a U-turn is present in the furthest nodes in the NUTS
        tree
        """
        dx = xplus - xminus
        return (np.dot(dx, rminus.T) < 0) or (np.dot(dx, rplus.T) < 0)

    def NUTSLeapfrog(self, x, r, grad_x, direction, step_size, temperature=1.0):
    
        """
        Description
        -----------
        Performs a single Leapfrog step returning the final position, momentum and gradient.
        """

        r = np.add(r, (direction*step_size/2)*grad_x)
        x = np.add(x, direction*step_size*np.multiply(self.inv_metric, r))
        
        grad_x = self.target.logpdfgrad(x, phi=temperature)
  
        r = np.add(r, (direction*step_size/2)*grad_x)

        #print(x)
        #print(r)
        #exit()
        return x, r, grad_x

    def logpdf(self, smc_state):
        """
        Description:
            Calculate the log probability of the forward kernel.

        Args:
            v: Particle velocities.

        Returns:
            log_prob: Log probability of the forward kernel.
        """

        return self.momentum_proposal.logpdf(smc_state.momenta_old)
    
    def set_inv_metric(self, diag_elements_as_array):
        self.inv_metric = diag_elements_as_array
    
    def set_step_size(self, step_size):
        self.step_size = step_size



class NUTSProposalAcceptReject(NUTSProposal):
    def __init__(
        self,
        target,
        momentum_proposal,
        step_size: float,
        rng = np.random.default_rng(),
    ):
        super().__init__(target, momentum_proposal, step_size, rng)
    
    def propose(self, smc_state):
        smc_state = super().propose(smc_state)

        accepted = hmc_accept_reject(
            smc_state, rng=self.rng,
        )

        smc_state.positions = np.where(accepted[:, np.newaxis], smc_state.positions, smc_state.positions_old)
        smc_state.momenta = np.where(accepted[:, np.newaxis], smc_state.momenta, smc_state.momenta_old)
        smc_state.log_pdfs = np.where(accepted, smc_state.log_pdfs, smc_state.log_pdfs_old)
        smc_state.log_pdf_grads = np.where(accepted[:, np.newaxis], smc_state.log_pdf_grads, smc_state.log_pdf_grads_old)

        return smc_state
