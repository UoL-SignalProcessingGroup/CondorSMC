import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from time import sleep, time

import autograd.numpy as np  # type: ignore
from autograd import elementwise_grad as egrad  # type: ignore
from autograd.scipy import stats as AutoStats  # type: ignore
from scipy.stats import multivariate_normal  # type: ignore

from . import definitions, utils
from .job_types import JobType, DaemonRole, DaemonStatus, JobStatus

from condorcmf.dbqueue.job import Job as DBQJob
from condorcmf.dbqueue.result import Result as DBQResult
from condorcmf.scheduler import utils as SchedulerUtils
from condorcmf.scheduler.job import Job as SchedulerJob
from .smcs import importance_sampling

logger = logging.getLogger(__name__)


def initialise_sampler(args, target):
    if args.lkernel == "pseudo":
        from .smcs.lkernel.forward_lkernel import ForwardLKernel

        lkernel = ForwardLKernel(dim=target.dim)
    elif args.lkernel == "gauss":
        from .smcs.lkernel.gaussian_lkernel import GaussianApproxLKernel

        lkernel = GaussianApproxLKernel(N=args.nsamples, dim=target.dim)
    else:
        lkernel = None

    if args.recycling == "ess":
        from condorsmc.smcs.recycling.ess import ESSRecycling

        dim = target.constrained_dim if hasattr(target, "constrained_dim") else target.dim
        recycling = ESSRecycling(K=args.niters, dim=dim)
    else:
        recycling = None

    if args.proposal == "rw":
        from condorsmc.smcs.proposal.random_walk import RandomWalkProposal

        forward_kernel = RandomWalkProposal(target.dim)

        return forward_kernel, lkernel, recycling
    elif args.proposal == "nuts" or args.proposal == "hmc":
        from condorsmc.smcs.integrator.leapfrog import LeapfrogIntegrator

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

        return (
            forward_kernel,
            momentum_proposal,
            integrator,
            lkernel,
            recycling,
        )


def sample(args, target, smcs_args, dbq_db, dbq_session, dbq_daemon, dbq_job, dbq_checkpoint, global_iter):
    last_update = time()

    # Initial samples
    x = np.array(dbq_job.payload["x"])
    x_new = np.zeros([args.nsamples, target.dim])

    # Calculate the initial weights
    logw = np.array(dbq_job.payload["logw"])
    logw_new = np.zeros(args.nsamples)
    if "p_logpdf_x" in dbq_job.payload:
        p_logpdf_x = np.array(dbq_job.payload["p_logpdf_x"])
    else:
        p_logpdf_x = target.logpdf(x)

    # Tensors to hold velocities, gradients and number of leapfrog steps
    if args.proposal == "nuts" or args.proposal == "hmc":
        v_new = np.zeros([args.nsamples, target.dim])
        grad_x = np.zeros([args.nsamples, target.dim])
        num_steps = np.zeros(args.nsamples)

    # Unpack the sampler arguments
    if args.proposal == "rw":
        forward_kernel, lkernel, recycling = smcs_args
    elif args.proposal == "nuts" or args.proposal == "hmc":
        (
            forward_kernel,
            momentum_proposal,
            integrator,
            lkernel,
            recycling,
        ) = smcs_args

    # Check if the deadline has passed
    if time() > dbq_job.deadline:
        logger.warning("Passed sampling deadline, aborting job")
        dbq_job.set_status(JobStatus.FAILED)
        return x, logw, p_logpdf_x

    logger.info("Sampling for %.2f seconds", dbq_job.deadline - time())

    # Run the job until the deadline
    _iter = 0
    start_time = time()
    # sleep_epsilon = 0.01
    mean_estimates = []
    variance_estimates = []
    ess_iters = []
    log_likelihoods = []
    recycling_constants = []
    while float(time()) - float(dbq_job.deadline) < 0:
        if (args.niters is not None and _iter >= args.niters) or (
            float(time()) > float(dbq_job.deadline)
        ):
            break

        if time() > dbq_session.deadline and _iter > 0:
            dbq_job.set_status(JobStatus.COMPLETE)
            dbq_daemon.set_status(DaemonStatus.IDLE)
            logger.info("Passed session deadline, aborting job")
            break
        elif time() > dbq_session.deadline:
            dbq_job.set_status(JobStatus.FAILED)
            dbq_daemon.set_status(DaemonStatus.IDLE)
            logger.info("Passed session deadline before first iteration, aborting job")
            return x, logw, p_logpdf_x

        if time() - last_update > 30:
            dbq_daemon.set_status(DaemonStatus.BUSY)
            last_update = time()

        # Normalise importance weights and calculate the log likelihood
        wn, log_likelihood = importance_sampling.normalise_weights(logw)
        log_likelihoods.append(log_likelihood)

        if recycling:
            # Calculate the recycling constant
            recycling_constant = recycling.constant(logw, wn)
            recycling_constants.append(recycling_constant)

            # Estimate the mean and variance of the target distribution
            mean_estimate, variance_estimate = importance_sampling.estimate(x, wn, target)
            mean_estimates.append(mean_estimate)
            variance_estimates.append(variance_estimate)

        # Calculate the effective sample size and resample if necessary
        ess = importance_sampling.calculate_ess(wn)
        logger.debug("ESS: %.4f", ess / args.nsamples)
        ess_iters.append(ess)
        if ess < args.nsamples / 2:
            x, logw = importance_sampling.resample(x, wn, log_likelihood)

        # Propogate samples through the forward kernel
        if args.proposal == "rw":
            x_new = forward_kernel.rvs(x, args.nsamples)
        elif args.proposal == "nuts" or args.proposal == "hmc":
            v = forward_kernel.momentum_proposal.rvs(args.nsamples)
            grad_x = target.logpdfgrad(x)
            x_new, v_new, num_steps = forward_kernel.rvs(x, v, grad_x)

        p_logpdf_x = target.logpdf(x)
        p_logpdf_xnew = target.logpdf(x_new)

        if lkernel and not (args.proposal == "rw" and args.lkernel == "pseudo"):
            # Evaluate the target distribution, l kernel and forward kernel
            if args.proposal == "rw":
                lkernel_logpdf = lkernel.calculate_rw(x, x_new)
                q_logpdf = forward_kernel.logpdf(x, x_new)
            elif args.proposal == "nuts" or args.proposal == "hmc":
                lkernel_logpdf = lkernel.calculate_hmc(x, x_new, v, v_new)
                q_logpdf = forward_kernel.logpdf(v)
            logw_new = logw + p_logpdf_xnew - p_logpdf_x + lkernel_logpdf - q_logpdf
        else:
            logw_new = logw + p_logpdf_xnew - p_logpdf_x

        # Ensure that all samples and log weights are finite
        if np.all(np.isnan(x_new)) or np.all(np.isinf(x_new)) or np.all(np.isnan(logw_new)) or np.all(np.isinf(logw_new)):
            logger.error("All samples or log weights are nan/inf, aborting job")
            dbq_job.set_status(JobStatus.FAILED)
            dbq_daemon.set_status(DaemonStatus.IDLE)
            # print(f"[{datetime.now()}] All samples or log weights are nan or inf, aborting job")
            return x, logw, p_logpdf_x

        # Update x, logw and iter
        x = x_new.copy()
        logw = logw_new.copy()
        _iter += 1

        # # There is a small chance that we exit the sampling job early
        # if np.random.rand() < sleep_epsilon:
        #     print("!!! Exiting sampling job early")
        #     break
        # sleep_epsilon *= 1.01

    # Estimate the mean and variance of the target distribution
    mean_estimate, variance_estimate = importance_sampling.estimate(x, wn, target)
    mean_estimates.append(mean_estimate)
    variance_estimates.append(variance_estimate)

    dbq_daemon.set_status(DaemonStatus.BUSY)
    dbq_job.set_status(JobStatus.COMPLETE)
    logger.info("Sampling complete after %d iterations (%d samples)", _iter, _iter * args.nsamples)

    if recycling:
        # Recycle the mean and variance estimates
        recycling.K = _iter
        recycled_mean_estimates = recycling.recycle_mean(
            np.array(mean_estimates), np.array(recycling_constants)
        )
        recycled_variance_estimates = recycling.recycle_variance(
            np.array(variance_estimates),
            np.array(mean_estimates),
            recycled_mean_estimates,
            np.array(recycling_constants),
        )

    # Store results in database
    results_payload = {
        "mean_estimates": mean_estimates[-1].tolist(),
        "var_estimates": variance_estimates[-1].tolist(),
        "ess": ess_iters[-1].tolist(),
        "log_likelihoods": log_likelihoods[-1].tolist(),
    }

    if recycling:
        results_payload["recycled_mean_estimates"] = recycled_mean_estimates.tolist()
        results_payload[
            "recycled_variance_estimates"
        ] = recycled_variance_estimates.tolist()


    results_attributes = {
        "node_id": args.node_id,
        "role": args.role,
        "origin_job_id": dbq_job.job_id,
        "run_time": time() - start_time,
        "global_iter": global_iter,
        "K": _iter,
        "N": args.nsamples,
    }

    result = DBQResult(
        db=dbq_db,
        session_id=dbq_session.session_id,
        node_id=args.node_id,
        role=DaemonRole.FOLLOWER,
        attributes=results_attributes,
        payload=results_payload,
    )

    result.insert()

    # Send results to coordinator or manager
    payload = {
        "x": x.tolist(),
        "logw": logw.tolist(),
        "p_logpdf_x": p_logpdf_x.tolist(),
    }

    if dbq_checkpoint.exists(0):
        dbq_checkpoint.update(0, payload)
    else:
        dbq_checkpoint.create(0, payload)

    # print(f"{datetime.now()} Creating response job")
    payload = {
        "origin_job_id": dbq_job.job_id,
        "K": _iter,
        "N": args.nsamples,
        "mean_estimates": mean_estimates[-1].tolist(),
        "var_estimates": variance_estimates[-1].tolist(),
        "ess": ess_iters[-1].tolist(),
        "log_likelihoods": log_likelihoods[-1].tolist(),
    }

    if recycling:
        payload["recycled_mean_estimates"] = recycled_mean_estimates[-1].tolist()
        payload["recycled_var_estimates"] = recycled_variance_estimates[-1].tolist()
        payload["recycling_constants"] = recycling_constants[-1].tolist()

    response_job = DBQJob(
        db=dbq_db,
        session_id=args.session_id,
        round_id=dbq_job.round_id,
        to_id=dbq_job.from_id,
        from_id=args.node_id,
        type=JobType.IMPORTANCE_SAMPLING_RESULT,
        deadline=time() + args.follower_runtime,
        payload=payload,
    )
    response_job.create()
    # print(f"Response job created with id {response_job.job_id}")

    dbq_job.set_status(JobStatus.COMPLETE)

    return x, logw, p_logpdf_x


def main(args):
    with open("job.json", "r") as f:
        session_args = json.load(f)

    target = utils.load_target(".", f"{args.model}")

    logger.info("Initialising database")
    dbq_db, dbq_session, dbq_daemon, dbq_checkpoint = utils.initialise_daemon(args, session_args)

    logger.info("Node took %.2f seconds to initialise", time() - session_args['start_time'])

    smcs_args = initialise_sampler(args, target)
    # Check for checkpoint
    if dbq_checkpoint.exists(0):
        logger.info("Checkpoint found, loading...")
        checkpoint_payload = dbq_checkpoint.get(0)
        smcs_x = np.array(checkpoint_payload["x"])
        smcs_logw = np.array(checkpoint_payload["logw"])
        smcs_p_logpdf_x = np.array(checkpoint_payload["p_logpdf_x"])
    else:
        logger.info("No checkpoint found, initialising...")
        sample_proposal = multivariate_normal(
            mean=np.zeros(target.dim), cov=np.eye(target.dim)
        )
        smcs_x = sample_proposal.rvs(args.nsamples)
        smcs_p_logpdf_x = target.logpdf(smcs_x)
        smcs_q0_logpdf_x = sample_proposal.logpdf(smcs_x)
        smcs_logw = smcs_p_logpdf_x - smcs_q0_logpdf_x

    _iter = 0
    _sampling_iter = 0
    last_job = time()
    logger.debug("Session deadline: %s", dbq_session.deadline)
    while time() < dbq_session.deadline:
        dbq_daemon.set_status(DaemonStatus.ACTIVE)
        _iter_start = time()

        logger.debug(
            "Sampling iteration %d | %.2f seconds remaining",
            _sampling_iter, dbq_session.deadline - time(),
        )

        # Get the next job from the queue
        dbq_job = dbq_daemon.fetch_job()
        if dbq_job:
            logger.info("Received job %s from %s", dbq_job.job_id, dbq_job.from_id)

            if dbq_job.type == JobType.IMPORTANCE_SAMPLING:
                logger.info("Importance sampling job received")

                dbq_daemon.set_status(DaemonStatus.BUSY)
                dbq_job.set_status(JobStatus.RUNNING)

                if (
                    "x" not in dbq_job.payload and "logw" not in dbq_job.payload
                ):
                    logger.debug("Resuming sampling from local state")
                    dbq_job.payload = {
                        "x": smcs_x,
                        "logw": smcs_logw,
                        "p_logpdf_x": smcs_p_logpdf_x,
                    }
                else:
                    logger.debug("Fetching samples from job payload")
                    dbq_job.get_payload()

                smcs_x, smcs_logw, smcs_p_logpdf_x = sample(
                    args, target, smcs_args, dbq_db, dbq_session, dbq_daemon, dbq_job, dbq_checkpoint, _sampling_iter
                )
                _sampling_iter += 1

                dbq_daemon.set_status(DaemonStatus.ACTIVE)

            last_job = time()

        # Sleep for the remainder of the tick rate
        # if time() - last_job > 60:
        #     print("No jobs received for 60 seconds, exiting")
        #     dbq_daemon.set_status(DaemonStatus.TERMINATED)
        #     return 0

        _iter += 1
        _iter_run_time = time() - _iter_start
        if _iter_run_time < definitions.CONDORSMC_TICK_RATE:
            dbq_daemon.set_status(DaemonStatus.IDLE)
            _iter_sleep_time = definitions.CONDORSMC_TICK_RATE - _iter_run_time
            # print(f"Sleeping for {_iter_sleep_time:2f} seconds")
            sleep(_iter_sleep_time)

    # Set status to inactive
    logger.info("Passed session deadline, aborting session")
    dbq_daemon.set_status(DaemonStatus.TERMINATED)

    return 0
