import json
import uuid
from datetime import datetime
from pathlib import Path
from time import sleep, time

import autograd.numpy as np  # type: ignore
from autograd import elementwise_grad as egrad  # type: ignore
from autograd.scipy import stats as AutoStats  # type: ignore
from scipy.stats import multivariate_normal  # type: ignore

from . import definitions, utils

from condorcmf.dbqueue.job import Job as DBQJob
from condorcmf.dbqueue.result import Result as DBQResult
from condorcmf.scheduler import utils as SchedulerUtils
from condorcmf.scheduler.job import Job as SchedulerJob
from .smcs import importance_sampling

RESULTS_JOB_CODES = [1]


def result_map(result_job_type):
    result_map = {
        1: 0,  # Results (job code 1) from importance sampling (job code 0)
    }

    return result_map.get(result_job_type, "-1")


def initialise_sampling_jobs(args, dbq_db, followers):
    round_id = str(uuid.uuid4())
    deadline = time() + args.follower_runtime
    jobs = []
    for follower in followers:
        job = DBQJob(
            db=dbq_db,
            session_id=args.session_id,
            round_id=round_id,
            to_id=follower,
            from_id=args.node_id,
            type=0,
            deadline=deadline,
        )
        job.create()
        jobs.append(job)
    
    return jobs, round_id


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

    if results_type == 1:
        dbq_daemon.set_status(5)

        mean_estimates = []
        var_estimates = []
        recycled_mean_estimates = []
        recycled_var_estimates = []
        log_likelihoods = []
        recycling_constants = []

        print(f"Processing {len(dbq_jobs)} importance sampling jobs")

        # Process results and schedule importance sampling jobs
        for dbq_job in dbq_jobs:
            dbq_job.set_status(1)
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
                type=0,
                session_id=args.session_id,
                job_id=results_payload["origin_job_id"],
            )

            origin_job.set_status(4)

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
        if args.resampling and len(dbq_jobs) > 1 and reschedule:
            print(f"[{datetime.now()}] Global resampling")
            active_nodes = [dbq_job.from_id for dbq_job in dbq_jobs]
            importance_sampling.checkpoint_resample(args, dbq_db, active_nodes)
        else:
            print(f"[{datetime.now()}] No global resampling")
            active_nodes = [dbq_job.from_id for dbq_job in dbq_jobs]

        # Reschedule importance sampling jobs
        deadline = time() + args.follower_runtime
        jobs = []
        for dbq_job in dbq_jobs:
            if reschedule:
                job = DBQJob(
                    db=dbq_db,
                    session_id=args.session_id,
                    round_id=next_round_id,
                    to_id=dbq_job.from_id,
                    from_id=args.node_id,
                    type=0,
                    deadline=deadline,
                )
                print(f"Scheduling importance sampling job {job.job_id}")
                job.create()
                jobs.append(job)
                dbq_job.set_status(4)
            else:
                print(f"Removing importance sampling job")
                dbq_job.delete()

        # Reschedule stale jobs
        for stale_job in stale_jobs:
            if reschedule:
                stale_job.get_payload()
                job = DBQJob(
                    db=dbq_db,
                    session_id=args.session_id,
                    round_id=next_round_id,
                    to_id=stale_job.to_id,
                    from_id=stale_job.from_id,
                    type=0,
                    deadline=deadline,
                    payload=stale_job.payload,
                )
                print(f"Rescheduling stale job {stale_job.job_id}")
                job.create()
                jobs.append(job)
                stale_job.set_status(4)
            else:
                print(f"Removing stale job")
                stale_job.delete()

        results_payload = {
            "mean_estimate": mean_estimate,
            "var_estimate": var_estimate,
            "z": z,
        }

        results_attributes = {
            "active_nodes": active_nodes,
            "sampling_iter": sampling_iter,
        }

        dbq_result = DBQResult(
            db=dbq_db,
            session_id=args.session_id,
            node_id=args.node_id,
            role=1,
            attributes=results_attributes,
            payload=results_payload,
        )

        dbq_result.insert()

        payload = {
            "active_nodes": active_nodes,
            "mean_estimate": np.array(mean_estimate),
            "var_estimate": np.array(var_estimate),
            "z": np.array(z),
        }

        dbq_daemon.set_status(1)

        return payload, jobs, next_round_id


def sample(args, followers, dbq_db, dbq_session, dbq_daemon, dbq_job):
    print(f"ORIGIN JOB ID {dbq_job.job_id}")
    last_update = time()

    # Check if the deadline has passed
    if time() > dbq_job.deadline:
        print(f"[{datetime.now()}] Passed sampling deadline, aborting job")
        dbq_job.set_status(3)
        dbq_daemon.set_status(0)
        return

    sampling_jobs, round_id = initialise_sampling_jobs(args, dbq_db, followers)
    last_round_id = round_id

    print(f"Sampling for {dbq_job.deadline - time()} seconds")

    # Run the job until the deadline
    _iter = 0
    start_time = time()
    _importance_sampling_iter = 0
    active_nodes = []
    mean_estimates = []
    var_estimates = []
    zs = []
    while float(time()) < float(dbq_job.deadline):
        dbq_daemon.set_status(1)
        _iter_start = time()
        print(
            f"[{datetime.now()}] Local sampling iteration {_importance_sampling_iter} "
            f"| {dbq_session.deadline - time():2f} seconds remaining"
        )

        # Check for stale followers
        dbq_session.clean_stale_daemons(ids=followers, timeout=definitions.CONDORSMC_FOLLOWER_TIMEOUT, role=1)

        # Number of active followers
        n_active_followers = dbq_session.n_active_daemons(ids=followers)
        print(f"[{datetime.now()}] Number of active followers: {n_active_followers}")

        if time() > dbq_session.deadline:
            for sampling_job in sampling_jobs:
                sampling_job.delete()
            if _importance_sampling_iter > 0:            
                print(f"[{datetime.now()}] Passed session deadline, exiting main sampling loop and sending results to coordinator")
                break
            dbq_job.set_status(3)
            dbq_daemon.set_status(0)
            print(f"[{datetime.now()}] Passed session deadline, exiting main sampling loop")
            return

        # Get the next job from the queue
        response_dbq_job = dbq_daemon.fetch_job(round_id=round_id)
        if response_dbq_job:
            origin_job = utils.fetch_origin_job(dbq_db, response_dbq_job)
            if response_dbq_job.type in RESULTS_JOB_CODES and time() + definitions.CONDORSMC_FOLLOWER_DEADLINE_BUFFER > origin_job.deadline:
                dbq_session.clean_stale_jobs(
                    job_type=result_map(response_dbq_job.type), check_deadline=False, from_id=args.node_id, clear_running=True
                )  # Set jobs that haven't started to stale

                # Number of active jobs
                n_active_jobs = dbq_session.n_active_jobs(
                    job_type=result_map(response_dbq_job.type),
                    from_id=args.node_id,
                )

                # Request all results
                response_dbq_jobs = dbq_daemon.fetch_all_jobs(job_type=response_dbq_job.type, round_id=round_id)
                print(
                    f"[{datetime.now()}] There are {n_active_jobs} active jobs and {len(response_dbq_jobs)} results"
                )

                # Fetch stale jobs from queue
                stale_jobs, n_stale_jobs = dbq_session.fetch_stale_jobs(
                    job_type=result_map(response_dbq_job.type), from_id=args.node_id
                )

                # Process results and handle stale jobs
                last_round_id = round_id
                payload, sampling_jobs, round_id = process_results(
                    args,
                    dbq_db,
                    dbq_session,
                    dbq_daemon,
                    1,
                    response_dbq_jobs,
                    stale_jobs,
                    _importance_sampling_iter
                )

                active_nodes = payload["active_nodes"]
                mean_estimates.append(payload["mean_estimate"])
                var_estimates.append(payload["var_estimate"])
                zs.append(payload["z"])

                _importance_sampling_iter += 1

                # Remove jobs pending deltion
                dbq_session.clean_jobs(round_id=last_round_id)

        # If followers take too long to initialise, the initial importance
        # sampling jobs may be stale. Check for stale jobs and reinitialise.
        elif _importance_sampling_iter == 0:
            stale_jobs, n_stale_jobs = dbq_session.fetch_stale_jobs(job_type=0, from_id=args.node_id)
            print(f"[{datetime.now()}] Number of stale jobs: {n_stale_jobs}")

            if n_stale_jobs > 0 and n_stale_jobs == args.nfollowers:
                # warnings.warn("All importance sampling jobs are stale. Reinitialising.")
                print(f"[{datetime.now()}] All importance sampling jobs are stale. Reinitialising.")
                for stale_job in stale_jobs:
                    stale_job.delete()
                sampling_jobs = initialise_sampling_jobs(
                    args, dbq_db, followers
                )

        _iter += 1
        _iter_run_time = time() - _iter_start
        if _iter_run_time < definitions.CONDORSMC_TICK_RATE:
            dbq_daemon.set_status(0)
            _iter_sleep_time = definitions.CONDORSMC_TICK_RATE - _iter_run_time
            print(f"[{datetime.now()}] Sleeping for {_iter_sleep_time:2f} seconds")
            sleep(_iter_sleep_time)

    # Number of active jobs
    n_active_jobs = dbq_session.n_active_jobs(
        job_type=0,
        from_id=args.node_id,
    )

    if n_active_jobs > 0 and not args.nowait:
        print("Deadline reached. Waiting for final results.")
        while True:
            _iter_start = time()
            print(
                f"[{datetime.now()}] Local sampling iteration {_importance_sampling_iter} "
                f"| {dbq_session.deadline - time():2f} seconds remaining"
            )

            # Check for stale followers
            dbq_session.clean_stale_daemons(ids=followers, timeout=definitions.CONDORSMC_FOLLOWER_TIMEOUT, role=1)

            # Number of active followers
            n_active_followers = dbq_session.n_active_daemons(ids=followers)
            print(f"[{datetime.now()}] Number of active followers: {n_active_followers}")

            # Get the next job from the queue
            # Exclude results from the follower nodes
            response_dbq_job = dbq_daemon.fetch_job(round_id=round_id)
            if response_dbq_job:
                origin_job = utils.fetch_origin_job(dbq_db, response_dbq_job)
                if response_dbq_job.type in RESULTS_JOB_CODES and time() + definitions.CONDORSMC_FOLLOWER_DEADLINE_BUFFER > origin_job.deadline:
                    dbq_session.clean_stale_jobs(
                        job_type=result_map(response_dbq_job.type), check_deadline=False, from_id=args.node_id, round_id=round_id, clear_running=True
                    )  # Set jobs that haven't started to stale

                    # Number of active jobs
                    n_active_jobs = dbq_session.n_active_jobs(
                        job_type=result_map(response_dbq_job.type),
                        from_id=args.node_id,
                    )

                    # Request all results
                    response_dbq_jobs = dbq_daemon.fetch_all_jobs(job_type=response_dbq_job.type)
                    print(
                        f"[{datetime.now()}] WAIT JOBS There are {n_active_jobs} active jobs and {len(response_dbq_jobs)} results"
                    )

                    # Fetch stale jobs from queue
                    stale_jobs, n_stale_jobs = dbq_session.fetch_stale_jobs(
                        job_type=result_map(response_dbq_job.type), from_id=args.node_id
                    )

                    # Process results and handle stale jobs
                    last_round_id = round_id
                    payload, sampling_jobs, round_id = process_results(
                        args,
                        dbq_db,
                        dbq_session,
                        dbq_daemon,
                        1,
                        response_dbq_jobs,
                        stale_jobs,
                        _importance_sampling_iter,
                        reschedule=False
                    )

                    active_nodes = payload["active_nodes"]
                    mean_estimates.append(payload["mean_estimate"])
                    var_estimates.append(payload["var_estimate"])
                    zs.append(payload["z"])

                    _importance_sampling_iter += 1

                    break

            _iter += 1
            _iter_run_time = time() - _iter_start
            if _iter_run_time < definitions.CONDORSMC_TICK_RATE:
                dbq_daemon.set_status(0)
                _iter_sleep_time = definitions.CONDORSMC_TICK_RATE - _iter_run_time
                print(f"[{datetime.now()}] Sleeping for {_iter_sleep_time:2f} seconds")
                sleep(_iter_sleep_time)

    else:
        dbq_session.clean_stale_jobs(job_type=0, check_deadline=False, from_id=args.node_id)
        stale_jobs, n_stale_jobs = dbq_session.fetch_stale_jobs(
            job_type=0, from_id=args.node_id
        )

        print(f"[{datetime.now()}] Number of stale jobs: {n_stale_jobs}")
        if n_stale_jobs == args.nfollowers:
            print(f"[{datetime.now()}] All jobs stale, deleting")
            for stale_job in stale_jobs:
                stale_job.delete()
            dbq_job.set_status(3)
            return

    # Remove jobs pending deltion
    dbq_session.clean_complete_jobs(to_id=args.node_id)
    dbq_session.clean_complete_jobs(from_id=args.node_id)

    dbq_job.set_status(2)

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

    payload = {
        "origin_job_id": dbq_job.job_id,
        "active_nodes": active_nodes,
        "mean_estimate": mean_estimate.tolist(),
        "var_estimate": var_estimate.tolist(),
        "z": z,
    }

    print(f"[{datetime.now()}] Sending results to coordinator, ", end="\r") # do not add newline
    response_job = DBQJob(
        db=dbq_db,
        session_id=args.session_id,
        round_id=dbq_job.round_id,
        to_id=dbq_job.from_id,
        from_id=args.node_id,
        type=3,
        deadline=time() + args.follower_runtime,
        payload=payload,
    )
    response_job.create()
    print(f"job id {response_job.job_id}")

    ## TO DO: Clean stray jobs from this round

    dbq_job.set_status(2)


def main(args):
    with open("job.json", "r") as f:
        session_args = json.load(f)

    target = utils.load_target(".", f"{args.model}")

    dbq_db, dbq_session, dbq_daemon, dbq_checkpoint = utils.initialise_daemon(args, session_args)

    print(f"Node took {time() - session_args['start_time']} seconds to initialise")

    _iter = 0
    _sampling_iter = 0
    last_job = time()
    while time() < dbq_session.deadline:
        dbq_daemon.set_status(1)
        _iter_start = time()

        print(
            f"[{datetime.now()} {dbq_session.deadline - time()}] Global sampling iteration {_sampling_iter}"
        )

        # Get the next job from the queue
        dbq_job = dbq_daemon.fetch_job(job_type=2)
        if dbq_job:
            print(
                f"{datetime.now()} Received job (type {dbq_job.type}) {dbq_job.job_id} from {dbq_job.from_id}"
            )
            if dbq_job.type == 2:
                print(f"Received job (type {dbq_job.type}) {dbq_job.job_id} from {dbq_job.from_id}")

                dbq_daemon.set_status(4)
                dbq_job.set_status(1)

                sample(args, session_args["followers"], dbq_db, dbq_session, dbq_daemon, dbq_job)
                _sampling_iter += 1

            last_job = time()

        # Sleep for the remainder of the tick rate
        # if time() - last_job > 60:
        #     print("No jobs received for 60 seconds, exiting")
        #     dbq_daemon.set_status(3)
        #     return 0

        _iter += 1
        _iter_run_time = time() - _iter_start
        if _iter_run_time < definitions.CONDORSMC_TICK_RATE:
            dbq_daemon.set_status(0)
            _iter_sleep_time = definitions.CONDORSMC_TICK_RATE - _iter_run_time
            print(f"Sleeping for {_iter_sleep_time:2f} seconds")
            sleep(_iter_sleep_time)

    # Set status to inactive
    print(f"[{datetime.now()}] Passed session deadline, aborting session")
    dbq_daemon.set_status(3)

    return 0
