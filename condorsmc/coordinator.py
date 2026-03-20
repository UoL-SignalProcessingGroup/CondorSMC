import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from time import sleep, time
from tqdm import tqdm

import autograd.numpy as np  # type: ignore
from autograd import elementwise_grad as egrad  # type: ignore
from autograd.scipy import stats as AutoStats  # type: ignore

from . import network, definitions, utils, writer
from .job_types import JobType, DaemonRole, DaemonStatus, JobStatus, RESULT_JOB_TYPES, RESULT_TO_ORIGIN

try:
    from .dbqueue.connector.mysql import MySQLConnector as DBQConnector  # type: ignore
except ImportError:
    from .dbqueue.connector.pymysql import PyMySQLConnector as DBQConnector  # type: ignore

logger = logging.getLogger(__name__)

from .dbqueue.daemon import Daemon as DBQDaemon
from .dbqueue.job import Job as DBQJob
from .dbqueue.result import Result as DBQResult
from .dbqueue.session import Session as DBQSession
from .scheduler import utils as SchedulerUtils
from .scheduler.job import Job as SchedulerJob
from .smcs import importance_sampling


# def clear_session(dbqueue_session, follower_scheduler_jobs):
def clear_session(dbqueue_session):
    # for scheduler_job in follower_scheduler_jobs:
    #     scheduler_job.clean()
    dbqueue_session.clear_session()


def initialise_sampling_jobs(args, dbq_db, dbq_session, cmf_daemons):
    round_id = str(uuid.uuid4())
    manager_deadline = time() + args.coordinator_runtime
    if args.nmanagers > 0:
        # Extract manager ids
        manager_ids = [manager["id"] for manager in cmf_daemons["managers"]]

        # Wait for manager nodes to register
        while dbq_session.n_active_daemons(role=DaemonRole.MANAGER, ids=manager_ids) == 0:
            logger.info("Waiting for manager nodes to register")
            sleep(15)
        
        deadline = time() + args.manager_runtime
        for manager in cmf_daemons["managers"]:
            job = DBQJob(
                db=dbq_db,
                session_id=args.session_id,
                round_id=round_id,
                to_id=manager["id"],
                from_id=args.node_id,
                type=JobType.MANAGER_SAMPLING,
                deadline=deadline,
            )
            job.create()
    else:
        # Extract follower ids
        follower_ids = [follower["id"] for follower in cmf_daemons["followers"]]

        # Wait for follower nodes to register
        while dbq_session.n_active_daemons(role=DaemonRole.FOLLOWER, ids=follower_ids) == 0:
            if time() > manager_deadline:
                raise RuntimeError("Coordinator deadline passed before any follower registered")
            logger.info("Waiting for follower nodes to register")
            sleep(15)

        deadline = time() + args.follower_runtime
        for follower in cmf_daemons["followers"]:
            job = DBQJob(
                db=dbq_db,
                session_id=args.session_id,
                round_id=round_id,
                to_id=follower["id"],
                from_id=args.node_id,
                type=JobType.IMPORTANCE_SAMPLING,
                deadline=deadline,
            )
            job.create()
    
    return round_id


def process_results(
    args,
    dbq_db,
    dbq_session,
    dbq_daemon,
    results_type,
    dbq_jobs,
    stale_jobs,
    sampling_iter,
    reschedule=True,
):
    next_round_id = str(uuid.uuid4())

    if results_type == JobType.IMPORTANCE_SAMPLING_RESULT:
        dbq_daemon.set_status(DaemonStatus.PROCESSING)

        mean_estimates = []
        var_estimates = []
        recycled_mean_estimates = []
        recycled_var_estimates = []
        log_likelihoods = []
        recycling_constants = []

        logger.debug("Scheduling %d importance sampling jobs", len(dbq_jobs))

        # Process results and schedule importance sampling jobs
        for dbq_job in dbq_jobs:
            dbq_job.set_status(JobStatus.RUNNING)
            results_payload = dbq_job.get_payload(store_payload=False)

            # Estimate quantities of interest
            mean_estimates.append(results_payload["mean_estimates"])
            var_estimates.append(results_payload["var_estimates"])
            log_likelihoods.append(results_payload["log_likelihoods"])
            if args.recycling is not None:
                recycled_mean_estimates.append(results_payload["recycled_mean_estimates"])
                recycled_var_estimates.append(results_payload["recycled_var_estimates"])
                recycling_constants.append(results_payload["recycling_constants"])
            
            # Mark origin importance sampling job for removal
            origin_job = DBQJob(
                db=dbq_db,
                to_id=dbq_job.from_id,
                from_id=args.node_id,
                type=JobType.IMPORTANCE_SAMPLING,
                session_id=args.session_id,
                job_id=results_payload["origin_job_id"],
            )

            origin_job.set_status(JobStatus.SUPERSEDED)

        # Convert to numpy arrays
        mean_estimates = np.array(mean_estimates)
        var_estimates = np.array(var_estimates)
        log_likelihoods = np.array(log_likelihoods)
        if args.recycling is not None:
            recycled_mean_estimates = np.array(recycled_mean_estimates)
            recycled_var_estimates = np.array(recycled_var_estimates)
            recycling_constants = np.array(recycling_constants)

        # Calculate mean and variance over all nodes
        if args.recycling == "ess":
            mean_estimate, var_estimate, z = importance_sampling.estimate_nodes(
                recycled_mean_estimates,
                recycled_var_estimates,
                recycling_constants,
            )
        else:
            mean_estimate, var_estimate, z = importance_sampling.estimate_nodes(
                mean_estimates,
                var_estimates,
                log_likelihoods,
            )

        # Global resampling
        if args.resampling and reschedule:
            logger.debug("Global resampling")
            active_nodes = [job.from_id for job in dbq_jobs]
            importance_sampling.checkpoint_resample(args, dbq_db, active_nodes)
        else:
            logger.debug("No global resampling")
            active_nodes = [dbq_job.from_id for dbq_job in dbq_jobs]

        # Reschedule importance sampling jobs
        deadline = time() + args.follower_runtime
        for dbq_job in dbq_jobs:
            if reschedule:
                logger.debug("Scheduling importance sampling job")
                job = DBQJob(
                    db=dbq_db,
                    session_id=args.session_id,
                    round_id=next_round_id,
                    to_id=dbq_job.from_id,
                    from_id=args.node_id,
                    type=JobType.IMPORTANCE_SAMPLING,
                    deadline=deadline,
                )
                job.create()
                dbq_job.set_status(JobStatus.SUPERSEDED)
            else:
                dbq_job.delete()

        # Reschedule stale jobs
        for stale_job in stale_jobs:
            if reschedule:
                logger.debug("Rescheduling stale job %s", stale_job.job_id)
                stale_job.get_payload()
                job = DBQJob(
                    db=dbq_db,
                    session_id=args.session_id,
                    round_id=next_round_id,
                    to_id=stale_job.to_id,
                    from_id=stale_job.from_id,
                    type=JobType.IMPORTANCE_SAMPLING,
                    deadline=deadline,
                )
                job.create()
                stale_job.set_status(JobStatus.SUPERSEDED)
            else:
                stale_job.delete()

        dbq_daemon.set_status(DaemonStatus.ACTIVE)

    if results_type == JobType.MANAGER_SAMPLING_RESULT:
        dbq_daemon.set_status(DaemonStatus.PROCESSING)

        logger.debug("Processing %d jobs", len(dbq_jobs))

        mean_estimates = []
        var_estimates = []
        zs = []

        for dbq_job in dbq_jobs:
            dbq_job.set_status(JobStatus.RUNNING)
            results_payload = dbq_job.get_payload()

            # Get results
            mean_estimates.append(np.array(results_payload["mean_estimate"]))
            var_estimates.append(np.array(results_payload["var_estimate"]))
            zs.append(np.array(results_payload["z"]))

            # Mark origin importance sampling job for removal
            origin_job = DBQJob(
                db=dbq_db,
                to_id=dbq_job.from_id,
                from_id=args.node_id,
                type=JobType.MANAGER_SAMPLING,
                session_id=args.session_id,
                job_id=results_payload["origin_job_id"],
            )

            origin_job.set_status(JobStatus.SUPERSEDED)

        # Convert to numpy arrays
        mean_estimates = np.array(mean_estimates)
        var_estimates = np.array(var_estimates)
        zs = np.array(zs)

        # Calculate mean and variance over all nodes
        mean_estimate, var_estimate, z = importance_sampling.estimate_nodes(
            mean_estimates,
            var_estimates,
            zs,
            logs=False,
        )

        # Global resampling
        if args.resampling and reschedule:
            logger.debug("Global resampling")
            active_nodes = []
            for dbq_job in dbq_jobs:
                active_nodes += dbq_job.payload["active_nodes"]
            importance_sampling.checkpoint_resample(args, dbq_db, active_nodes)

        # Schedule importance sampling jobs
        deadline = time() + args.manager_runtime
        for dbq_job in dbq_jobs:
            if reschedule:
                job = DBQJob(
                    db=dbq_db,
                    session_id=args.session_id,
                    round_id=next_round_id,
                    to_id=dbq_job.from_id,
                    from_id=args.node_id,
                    type=JobType.MANAGER_SAMPLING,
                    deadline=deadline,
                )
                job.create()
                dbq_job.set_status(JobStatus.SUPERSEDED)
            else:
                dbq_job.delete()

        # Reschedule stale jobs
        for stale_job in stale_jobs:
            if reschedule:
                logger.debug("Rescheduling stale job %s", stale_job.job_id)
                stale_job.get_payload()
                job = DBQJob(
                    db=dbq_db,
                    session_id=args.session_id,
                    round_id=next_round_id,
                    to_id=stale_job.to_id,
                    from_id=stale_job.from_id,
                    type=JobType.MANAGER_SAMPLING,
                    deadline=deadline,
                    payload=stale_job.payload,
                )
                job.create()
                stale_job.set_status(JobStatus.SUPERSEDED)
            else:
                stale_job.delete()

        dbq_daemon.set_status(DaemonStatus.ACTIVE)

    return mean_estimate, var_estimate, z, next_round_id


def main(args):
    target = utils.load_target(args.model_dir, args.model)

    # Initiliase CondorSMC session
    dbq_db, dbq_session, dbq_daemon, dbq_checkpoint = utils.initialise_daemon(args)

    if args.network_structure is not None:
        logger.info("Initialising CondorSMC network from YAML file")
        struct_path = Path(args.network_structure)
        if not struct_path.is_file():
            raise ValueError(f"Network structure file {args.network_structure} not found")

        session_daemons = network.load_network(
            args, dbq_session.deadline, struct_path
        )
    elif args.nmanagers > 0:
        logger.info("Initialising CondorSMC CMF network")
        # Initialise CondorSMC coordinator, manager and follower daemons
        session_daemons = network.initialise_cmf_daemons(
            args, dbq_session.deadline
        )
    else:
        logger.info("Initialising CondorSMC CF network")
        # Initialise CondorSMC coordinator and follower daemons
        session_daemons = network.initialise_cf_daemons(
            args, dbq_session.deadline
        )

    # Initialise sampling jobs
    round_id = initialise_sampling_jobs(args, dbq_db, dbq_session, session_daemons)
    last_round_id = round_id

    logger.info("Initialised with round id %s", round_id)

    # Arrays to store estimates from IS
    mean_estimates = []
    var_estimates = []
    constants = []

    # Main work loop
    _iter = 0
    _importance_sampling_iter = 0
    _total_runtime = int(dbq_session.deadline - time())
    with tqdm(
        total=_total_runtime,
        desc=f"[Iter {_importance_sampling_iter}] Sampling", unit="sec",
        bar_format="{desc} {percentage:3.0f}%|{bar}| {remaining}",
    ) as pbar:
        while time() < dbq_session.deadline:
            dbq_daemon.set_status(DaemonStatus.ACTIVE)
            _iter_start = time()
            logger.debug(
                "Sampling iteration %d | %.2f seconds remaining",
                _importance_sampling_iter,
                dbq_session.deadline - time(),
            )

            # Check for stale followers
            if args.nmanagers == 0:
                dbq_session.clean_stale_daemons(timeout=definitions.CONDORSMC_FOLLOWER_TIMEOUT, role=DaemonRole.FOLLOWER)
            # else:
            #     dbq_session.clean_stale_daemons(timeout=definitions.CONDORSMC_MANAGER_TIMEOUT, role=DaemonRole.MANAGER)

            # Number of active followers
            if args.nmanagers > 0:
                n_active_managers = dbq_session.n_active_daemons(role=DaemonRole.MANAGER)
            n_active_followers = dbq_session.n_active_daemons()

            # Get the next job from the queue
            dbq_job = dbq_daemon.fetch_job(round_id=round_id)
            if dbq_job:
                # If results, check that the deadline plus a buffer has been reached
                origin_job = utils.fetch_origin_job(dbq_db, dbq_job)
                if dbq_job.type in RESULT_JOB_TYPES and (
                    (time() + definitions.CONDORSMC_FOLLOWER_DEADLINE_BUFFER > origin_job.deadline and args.nmanagers == 0)
                    or (time() + definitions.CONDORSMC_MANAGER_DEADLINE_BUFFER > origin_job.deadline and args.nmanagers > 0)
                ):
                    dbq_session.clean_stale_jobs(
                        job_type=RESULT_TO_ORIGIN.get(dbq_job.type), check_deadline=False, from_id=args.node_id, round_id=round_id, clear_running=True
                    )  # Set jobs that haven't started or are still running to stale

                    # Number of active jobs
                    n_active_jobs = dbq_session.n_active_jobs(
                        job_type=RESULT_TO_ORIGIN.get(dbq_job.type),
                        from_id=args.node_id,
                        round_id=round_id,
                    )

                    # Request all results
                    dbq_jobs = dbq_daemon.fetch_all_jobs(job_type=dbq_job.type, round_id=round_id)
                    logger.debug(
                        "There are %d active jobs and %d results",
                        n_active_jobs, len(dbq_jobs),
                    )

                    # Fetch stale jobs from queue
                    stale_jobs, n_stale_jobs = dbq_session.fetch_stale_jobs(
                        job_type=RESULT_TO_ORIGIN.get(dbq_job.type), from_id=args.node_id, round_id=round_id
                    )

                    # Process results and handle stale jobs
                    last_round_id = round_id
                    mean_estimate, var_estimate, z, round_id = process_results(
                        args,
                        dbq_db,
                        dbq_session,
                        dbq_daemon,
                        dbq_job.type,
                        dbq_jobs,
                        stale_jobs,
                        _importance_sampling_iter,
                    )

                    results_payload = {
                        "mean_estimate": mean_estimate,
                        "var_estimate": var_estimate,
                        "z": z,
                    }

                    results_attributes = {
                        "sampling_iter": _importance_sampling_iter,
                        "active_followers": n_active_followers,
                    }

                    if args.nmanagers > 0:
                        results_attributes["active_managers"] = n_active_managers

                    dbq_result = DBQResult(
                        db=dbq_db,
                        session_id=args.session_id,
                        node_id=args.node_id,
                        role=DaemonRole.MANAGER,
                        attributes=results_attributes,
                        payload=results_payload,
                    )

                    dbq_result.insert()

                    mean_estimates.append(mean_estimate)
                    var_estimates.append(var_estimate)
                    constants.append(z)

                    _importance_sampling_iter += 1

                    # Remove jobs pending deltion
                    dbq_session.clean_jobs(round_id=last_round_id)

            # If followers take too long to initialise, the initial importance
            # sampling jobs may be stale. Check for stale jobs and reinitialise.
            elif args.nmanagers == 0:
                stale_jobs, n_stale_jobs = dbq_session.fetch_stale_jobs(job_type=JobType.IMPORTANCE_SAMPLING)
                logger.debug("Number of stale jobs: %d", n_stale_jobs)

                if n_stale_jobs == args.nfollowers:
                    if _importance_sampling_iter == 0:
                        # warnings.warn("All importance sampling jobs are stale. Reinitialising.")
                        logger.warning("All follower importance sampling jobs are stale. Reinitialising.")
                        for stale_job in stale_jobs:
                            stale_job.delete()
                        round_id = initialise_sampling_jobs(
                            args, dbq_db, dbq_session, session_daemons
                        )
                    else:
                        # resubmit stale jobs
                        logger.debug("Resubmitting stale jobs")
                        for stale_job in stale_jobs:
                            stale_job.get_payload()
                            job = DBQJob(
                                db=dbq_db,
                                session_id=args.session_id,
                                round_id=round_id,
                                to_id=stale_job.to_id,
                                from_id=args.node_id,
                                type=JobType.IMPORTANCE_SAMPLING,
                                deadline=stale_job.deadline,
                                payload=stale_job.payload,
                            )
                            job.create()
                            stale_job.delete()

            elif _importance_sampling_iter == 0 and args.nmanagers > 0:
                dbq_session.clean_stale_jobs(
                    job_type=JobType.MANAGER_SAMPLING, from_id=args.node_id, round_id=round_id
                )


                stale_jobs, n_stale_jobs = dbq_session.fetch_stale_jobs(job_type=JobType.MANAGER_SAMPLING)

                if n_stale_jobs == args.nmanagers:
                    # warnings.warn("All importance sampling jobs are stale. Reinitialising.")
                    logger.warning("All manager importance sampling jobs are stale. Reinitialising.")
                    for stale_job in stale_jobs:
                        stale_job.delete()
                    round_id = initialise_sampling_jobs(
                        args, dbq_db, dbq_session, session_daemons
                    )

            # Sleep for the remainder of the tick rate
            _iter += 1
            _iter_iter_run_time = time() - _iter_start
            if _iter_iter_run_time < definitions.CONDORSMC_TICK_RATE:
                dbq_daemon.set_status(DaemonStatus.IDLE)
                _iter_sleep_time = definitions.CONDORSMC_TICK_RATE - _iter_iter_run_time
                logger.debug("Sleeping for %.2f seconds", _iter_sleep_time)
                sleep(_iter_sleep_time)

            # Update progress bar
            _elapsed_seconds = int(time() - (dbq_session.deadline - _total_runtime))
            pbar.update(_elapsed_seconds - pbar.n)
            if args.nmanagers > 0:
                pbar.set_description(
                    f"[Iter {_importance_sampling_iter} | nmanagers={n_active_managers} | nfollowers={n_active_followers}] Sampling"
                )
            else:
                pbar.set_description(f"[Iter {_importance_sampling_iter} | nfollowers={n_active_followers}] Sampling")

    logger.info("Deadline reached. Waiting for final results.")
    # Check for active jobs
    n_active_jobs = dbq_session.n_active_jobs(from_id=args.node_id)
    if n_active_jobs > 0 and not args.nowait:
        while True:
            _iter_start = time()

            # Get the next job from the queue
            # Exclude results from the follower nodes
            dbq_job = dbq_daemon.fetch_job(round_id=round_id)
            if dbq_job:
                # If results, check that the deadline plus a buffer has been reached
                origin_job = utils.fetch_origin_job(dbq_db, dbq_job)
                if dbq_job.type in RESULT_JOB_TYPES and (
                    (time() + definitions.CONDORSMC_FOLLOWER_DEADLINE_BUFFER > origin_job.deadline and args.nmanagers == 0)
                    or (time() + definitions.CONDORSMC_MANAGER_DEADLINE_BUFFER > origin_job.deadline and args.nmanagers > 0)
                ):
                    dbq_session.clean_stale_jobs(
                        job_type=RESULT_TO_ORIGIN.get(dbq_job.type), check_deadline=False, from_id=args.node_id, round_id=round_id, clear_running=True
                    )  # Set jobs that haven't started or are still running to stale

                    # Number of active jobs
                    n_active_jobs = dbq_session.n_active_jobs(
                        job_type=RESULT_TO_ORIGIN.get(dbq_job.type),
                        from_id=args.node_id,
                        round_id=round_id,
                    )

                    # Request all results
                    dbq_jobs = dbq_daemon.fetch_all_jobs(job_type=dbq_job.type, round_id=round_id)
                    logger.debug(
                        "There are %d active jobs and %d results",
                        n_active_jobs, len(dbq_jobs),
                    )

                    # Fetch stale jobs from queue
                    stale_jobs, n_stale_jobs = dbq_session.fetch_stale_jobs(
                        job_type=RESULT_TO_ORIGIN.get(dbq_job.type), from_id=args.node_id, round_id=round_id
                    )

                    # Process results and handle stale jobs
                    mean_estimate, var_estimate, z, round_id = process_results(
                        args,
                        dbq_db,
                        dbq_session,
                        dbq_daemon,
                        dbq_job.type,
                        dbq_jobs,
                        stale_jobs,
                        _importance_sampling_iter,
                        reschedule=False,
                    )

                    results_payload = {
                        "mean_estimate": mean_estimate,
                        "var_estimate": var_estimate,
                        "z": z,
                    }

                    results_attributes = {
                        "sampling_iter": _importance_sampling_iter,
                        "active_followers": n_active_followers,
                    }

                    if args.nmanagers > 0:
                        results_attributes["active_managers"] = n_active_managers

                    dbq_result = DBQResult(
                        db=dbq_db,
                        session_id=args.session_id,
                        node_id=args.node_id,
                        role=DaemonRole.MANAGER,
                        attributes=results_attributes,
                        payload=results_payload,
                    )

                    dbq_result.insert()

                    mean_estimates.append(mean_estimate)
                    var_estimates.append(var_estimate)
                    constants.append(z)

                    _importance_sampling_iter += 1
                    break
            else:
                if args.nmanagers > 0:
                    dbq_session.clean_stale_jobs(job_type=JobType.MANAGER_SAMPLING, check_deadline=False, from_id=args.node_id)
                    stale_jobs, n_stale_jobs = dbq_session.fetch_stale_jobs(
                        job_type=JobType.MANAGER_SAMPLING, from_id=args.node_id
                    )

                    logger.debug("Number of stale jobs: %d", n_stale_jobs)
                    if n_stale_jobs == args.nmanagers:
                        logger.info("All jobs stale, deleting")
                        for stale_job in stale_jobs:
                            stale_job.delete()
                        break
                else:
                    dbq_session.clean_stale_jobs(job_type=JobType.IMPORTANCE_SAMPLING, check_deadline=False, from_id=args.node_id)
                    stale_jobs, n_stale_jobs = dbq_session.fetch_stale_jobs(
                        job_type=JobType.IMPORTANCE_SAMPLING, from_id=args.node_id
                    )

                    logger.debug("Number of stale jobs: %d", n_stale_jobs)
                    if n_stale_jobs == args.nfollowers:
                        logger.info("All jobs stale, deleting")
                        for stale_job in stale_jobs:
                            stale_job.delete()
                        break

            # Sleep for the remainder of the tick rate
            _iter_iter_run_time = time() - _iter_start
            if _iter_iter_run_time < definitions.CONDORSMC_TICK_RATE:
                dbq_daemon.set_status(DaemonStatus.IDLE)
                _iter_sleep_time = (
                    definitions.CONDORSMC_TICK_RATE - _iter_iter_run_time
                )
                logger.debug("Sleeping for %.2f seconds", _iter_sleep_time)
                sleep(_iter_sleep_time)

    logger.info("Processing final results from followers")
    mean_estimates = np.array(mean_estimates)
    var_estimates = np.array(var_estimates)

    if args.recycling == "ess":
        from .smcs.recycling.ess import ESSRecycling

        recycling = ESSRecycling(mean_estimates.shape[0], mean_estimates.shape[1])
    else:
        recycling = None

    logger.info("Writing results to output directory")
    if recycling is None:
        writer.write_results(
            args,
            definitions.SESSION_OUTPUT_DIR(args.session_id),
            mean_estimates,
            var_estimates,
        )
    else:
        recycled_mean_estimates = recycling.recycle_mean(mean_estimates, constants)
        recycled_var_estimates = recycling.recycle_variance(var_estimates, mean_estimates, recycled_mean_estimates, constants)
        writer.write_results(
            args,
            definitions.SESSION_OUTPUT_DIR(args.session_id),
            mean_estimates,
            var_estimates,
            recycled_mean_estimates,
            recycled_var_estimates,
        )

    # Clean up
    if args.session_id != "test":
        # clear_session(dbq_session, follower_scheduler_jobs)
        clear_session(dbq_session)

    logger.info(
        "CondorSMC session complete after %d global sampling iterations",
        _importance_sampling_iter,
    )
    logger.info(
        "Outputs from this session can be found in %s/%s",
        definitions.CONDORSMC_OUTPUT_DIR, args.session_id,
    )
