import importlib
import json
import shutil
import sys
from pathlib import Path
from time import time

from . import definitions, utils

try:
    from condorcmf.dbqueue.connector.mysql import MySQLConnector as DBQConnector  # type: ignore
except:
    from condorcmf.dbqueue.connector.pymysql import PyMySQLConnector as DBQConnector  # type: ignore

from condorcmf.dbqueue.checkpoint import Checkpoint as DBQCheckpoint
from condorcmf.dbqueue.daemon import Daemon as DBQDaemon
from condorcmf.dbqueue.job import Job as DBQJob
from condorcmf.dbqueue.session import Session as DBQSession


def initialise_daemon(args, json_args=None):
    if args.role == "coordinator":
        role = 0
    elif args.role == "manager":
        role = 1
    elif args.role == "follower":
        role = 2   
    else:
        raise ValueError(f"Invalid role {args.role}")

    # Create the database connection
    dbq_db = DBQConnector(
        host=definitions.MYSQL_HOST,
        user=definitions.MYSQL_USER,
        password=definitions.MYSQL_PASSWORD,
        database=definitions.MYSQL_DATABASE,
        poll_delay=definitions.MYSQL_POLL_DELAY,
    )

    # Initialise and register the dbqueue session
    if json_args is not None:
        dbq_session = DBQSession(
            db=dbq_db,
            session_id=args.session_id,
            deadline=json_args["session_deadline"],
        )
    else:
        dbq_session = DBQSession(
            db=dbq_db, session_id=args.session_id, runtime=args.coordinator_runtime + definitions.SESSION_DEADLINE_BUFFER
        )

        dbq_session.create()

    if args.role == "coordinator" and args.session_id == "test":
        dbq_db.delete("job_queue", f"`session_id`='{args.session_id}'", queue_query=True)
        dbq_db.delete("pool", f"`session_id`='{args.session_id}'", queue_query=True)
        dbq_db.delete("checkpoint", f"`session_id`='{args.session_id}'", queue_query=True)
        dbq_db.delete("results", f"`session_id`='{args.session_id}'", queue_query=True)
        dbq_db._execute_query_queue()

    # Initialise and register the dbqueue daemonl
    dbq_daemon = DBQDaemon(dbq_db, args.session_id, role, node_id=args.node_id)
    dbq_daemon.join()
    dbq_daemon.set_status(0)

    # Initialise the checkpoint
    dbq_checkpoint = DBQCheckpoint(
        dbq_db,
        args.session_id,
        args.node_id,
        role,
    )

    return dbq_db, dbq_session, dbq_daemon, dbq_checkpoint


def fetch_origin_job(dbq_db, dbq_job):
    """
    Fetch the origin job from the database
    
    Args:
        dbq_db (DBQConnector): Database connector
        dbq_job (DBQJob): Job to fetch the origin of

    Returns:
        DBQJob: Origin job
    """

    origin_id = dbq_job.get_payload(store_payload=False)["origin_job_id"]
    where_clause = f"session_id = '{dbq_job.session_id}' AND job_id = '{origin_id}'"
    job = dbq_db.select_one(
        "job_queue",
        "`id`, `session_id`, `job_id`, `round_id`, `to_id`, `from_id`, `type`, `created_at`, `deadline`",
        where_clause,
    )
    origin_job = DBQJob(
        db=dbq_db,
        session_id=job[1],
        to_id=job[4],
        from_id=job[5],
        type=job[6],
        job_id=job[2],
        round_id=job[3],
        created_at=job[7],
        deadline=job[8],
    )

    return origin_job


def load_target(model_dir, model_name):
    """
    Loads the target model from file, which can be either a Python file or a Stan file.

    Args:
        model_dir (str): Path to the directory containing the model file.
        model_name (str): Name of the model file.

    Returns:
        Target: Target model.
    """

    start = time()
    # Check if Python model file exists
    if Path(f"{model_dir}/{model_name}.py").exists():
        print(f"Found Python model file {model_name}.py")
        try:
            sys.path.append(model_dir)
            module = importlib.import_module(model_name)
            target_class = getattr(module, "Target")

            data_file = Path(f"{model_dir}/{model_name}.json")
            if data_file.exists():
                with data_file.open("r") as f:
                    data = json.load(f)
                target = target_class(data)
            else:
                target = target_class()

            return target
        except ImportError:
            raise ImportError(f"Failed to import target class from {model_name}.py")
    elif Path(f"{model_dir}/{model_name}.stan").exists():
        print(f"Found Stan model file {model_name}.stan")

        from condorsmc.smcs.target import StanModel

        # Check if data file exists
        data_file = Path(f"{model_dir}/{model_name}.json")

        if not data_file.exists():
            print(f"Could not find data file {model_name}.json")
            data_file = None

        target = StanModel(model_name, f"{model_dir}/{model_name}.stan", str(data_file))

    else:
        raise FileNotFoundError(
            f"Could not find model file {model_name}.py or {model_name}.stan"
        )

    print(f"Loaded target model in {time() - start:.2f} seconds")

    return target


def write_session_args(
    args, node_id, output_dir, coordinator_deadline, nfollowers, role="follower"
):
    args_list = [
        ("mode", str(args.mode)),
        ("session_id", str(args.session_id)),
        ("node_id", str(node_id)),
        ("role", str(role)),
        ("coordinator_runtime", str(coordinator_deadline)),
        ("manager_runtime", str(args.manager_runtime)),
        ("follower_runtime", str(args.follower_runtime)),
        ("nfollowers", str(nfollowers)),
        ("model_dir", str(args.model_dir)),
        ("model", str(args.model)),
        ("nsamples", str(args.nsamples)),
        ("niters", str(args.niters)),
        ("proposal", str(args.proposal)),
        ("lkernel", str(args.lkernel)),
        ("recycling", str(args.recycling)),
        ("integrator", str(args.integrator)),
        ("step_size", str(args.step_size)),
        ("hmc_steps", str(args.hmc_steps)),
        ("seed", str(args.seed)),
        ("verbose", str(args.verbose)),
    ]

    # Write the session arguments to file
    with open(Path(output_dir, "session_args.txt"), "w") as f:
        for arg in args_list:
            f.write(f"{arg[0]} {arg[1]}\n")


def archive_condorsmc(path, archive_path):
    """
    Archives a directory.

    Args:
        path (str): Path to the directory to be archived.
        archive_path (str): Path to the archive file.
    """

    shutil.make_archive(archive_path, "zip", path)
