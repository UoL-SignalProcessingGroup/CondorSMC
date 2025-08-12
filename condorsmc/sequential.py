import autograd.numpy as np  # type: ignore
from autograd import elementwise_grad as egrad  # type: ignore
from autograd.scipy import stats as AutoStats  # type: ignore
from scipy.stats import multivariate_normal  # type: ignore

from . import definitions, utils, writer

"""
TO DO
    - Add docstrings
"""


def main(args):
    target = utils.load_target(args.model_dir, args.model)

    sample_proposal = multivariate_normal(
        mean=np.zeros(target.dim), cov=np.eye(target.dim)
    )

    if args.lkernel == "pseudo":
        from .smcs.lkernel.forward_lkernel import ForwardLKernel

        lkernel = ForwardLKernel(dim=target.dim)
    elif args.lkernel == "gauss":
        from .smcs.lkernel.gaussian_lkernel import GaussianApproxLKernel

        lkernel = GaussianApproxLKernel(N=args.nsamples, dim=target.dim)
    else:
        lkernel = None

    if args.recycling == "ess":
        from .smcs.recycling.ess import ESSRecycling

        dim = target.constrained_dim if hasattr(target, "constrained_dim") else target.dim
        recycling = ESSRecycling(K=args.niters, dim=dim)
    else:
        recycling = None

    if args.proposal == "rw":
        from .smcs.proposal.random_walk import RandomWalkProposal
        from .smcs.smc_sampler_rw import RWSMCSampler

        forward_kernel = RandomWalkProposal(target.dim)
        smcs = RWSMCSampler(
            K=args.niters,
            N=args.nsamples,
            dim=target.dim,
            target=target,
            forward_kernel=forward_kernel,
            sample_proposal=sample_proposal,
            lkernel=lkernel,
            recycling=recycling,
            verbose=args.verbose,
            seed=args.seed,
        )

    elif args.proposal == "nuts" or args.proposal == "hmc":
        from .smcs.integrator.leapfrog import LeapfrogIntegrator
        from .smcs.smc_sampler_hmc import HMCSMCSampler

        momentum_proposal = multivariate_normal(
            mean=np.zeros(target.dim), cov=np.eye(target.dim)
        )
        integrator = LeapfrogIntegrator(target=target, step_size=args.step_size)

        if args.proposal == "nuts":
            from .smcs.proposal.nuts import NUTSProposal

            forward_kernel = NUTSProposal(
                target=target,
                integrator=integrator,
                momentum_proposal=momentum_proposal,
            )
        else:
            from .smcs.proposal.hmc import HMCProposal

            forward_kernel = HMCProposal(
                dim=target.dim,
                target=target,
                num_steps=args.hmc_steps,
                momentum_proposal=momentum_proposal,
                integrator=integrator,
            )

        smcs = HMCSMCSampler(
            K=args.niters,
            N=args.nsamples,
            dim=target.dim,
            target=target,
            forward_kernel=forward_kernel,
            sample_proposal=sample_proposal,
            lkernel=lkernel,
            recycling=recycling,
            verbose=args.verbose,
            seed=args.seed,
        )

    smcs.sample()

    print("Writing results to output directory")
    if recycling is None:
        writer.write_results(
            args,
            definitions.SESSION_OUTPUT_DIR(args.session_id),
            smcs.mean_estimate,
            smcs.variance_estimate,
        )
    else:
        writer.write_results(
            args,
            definitions.SESSION_OUTPUT_DIR(args.session_id),
            smcs.mean_estimate,
            smcs.variance_estimate,
            smcs.recycled_mean_estimate,
            smcs.recycled_variance_estimate,
        )

    print(f"-------------------------")
    print(
        f"Outputs from this session can be found in {definitions.CONDORSMC_OUTPUT_DIR}/{args.session_id}"
    )