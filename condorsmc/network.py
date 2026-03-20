import logging
import yaml
from pathlib import Path
from time import sleep, time

import uuid

from . import definitions, utils
from .scheduler.job import Job as SchedulerJob

logger = logging.getLogger(__name__)

PYTHON_ENV = definitions.PYTHON_ENV

def print_network_structure(daemons):
    """Log the network structure of the daemons."""
    lines = ["Network structure:", f"  Coordinator: {daemons['coordinator']['id']}"]
    if "managers" in daemons:
        lines.append("  Managers:")
        for manager in daemons["managers"]:
            lines.append(f"    id: {manager['id']}  followers: {manager['followers']}")
    lines.append("  Followers:")
    for follower in daemons["followers"]:
        lines.append(f"    id: {follower['id']}  manager: {follower.get('manager', '-')}")
    logger.info("\n".join(lines))


def initialise_cf_daemons(args, coordinator_deadline):
    # Load the follower config file
    if Path(Path.cwd(), "config", "follower_config.cfg").is_file():
        follower_config_path = Path(Path.cwd(), "config", "follower_config.cfg")
    elif Path(definitions.PACKAGE_CONFIG_DIR, "follower_config.cfg").is_file():
        follower_config_path = Path(
            definitions.PACKAGE_CONFIG_DIR, "follower_config.cfg"
        )
    else:
        raise FileNotFoundError("Could not find follower_config.cfg")

    # Archive the CondorSMC package
    utils.archive_condorsmc(
        definitions.PACKAGE_ROOT_DIR.parent,
        Path(
            definitions.SESSION_CACHE_DIR(args.session_id),
            "condorsmc",
        ),
    )

    # Archive the CondorCMF package
    utils.archive_condorsmc(
        CondorCMFDefinitions.PACKAGE_ROOT_DIR.parent,
        Path(
            definitions.SESSION_CACHE_DIR(args.session_id),
            "condorcmf",
        ),
    )

    cf_daemons = {
        "coordinator": {
            "id": args.node_id,
        },
        "followers": [],
    }

    executable_path = Path(definitions.PACKAGE_ROOT_DIR, "bin", "daemon.bat")

    # Initialise follower daemons
    start = time()
    for i in range(args.nfollowers):
        # pad i with zeros
        follower_id = f"test_follower_{str(i).zfill(4)}" if args.session_id == "test" else str(uuid.uuid4())
        cf_daemon = {
            "id": follower_id,
        }
        
        payload = {
            "session_id": args.session_id,
            "start_time": start,
            "session_deadline": coordinator_deadline,
            "coordinator_id": args.node_id,
            "job_id": follower_id,
        }

        cf_daemon["scheduler_job"] = SchedulerJob(
            executable=executable_path,
            requests=[("request_cpus", 1)],
            requirements='(Arch=="X86_64") && (OpSys=="Windows")',
            # requirements='(Arch=="X86_64") && (OpSys=="Windows") && (DA_CDT_HOST==TRUE)',
            # requirements='(Arch=="X86_64") && (OpSys=="LINUX") && (DA_CDT_HOST==TRUE)',
            # custom_args=["+DACDTJob = true"],
            session_id=args.session_id,
            job_id=follower_id,
            payload=payload,
        )

        input_files = [
            Path(
                definitions.SESSION_CACHE_DIR(args.session_id),
                "condorsmc.zip",
            ),
            Path(
                definitions.SESSION_CACHE_DIR(args.session_id),
                "condorcmf.zip",
            ),
            PYTHON_ENV,
            follower_config_path,
            Path(cf_daemon["scheduler_job"]._cache_dir, "session_args.txt"),
        ]

        if Path(args.model_dir, f"{args.model}.py").is_file():
            model_path = Path(args.model_dir, f"{args.model}.py")
        elif Path(args.model_dir, f"{args.model}.stan").is_file():
            model_path = Path(args.model_dir, f"{args.model}.stan")
        model_data = Path(args.model_dir, f"{args.model}.json")

        input_files.append(model_path)
        if model_data.is_file():
            input_files.append(model_data)

        cf_daemon["scheduler_job"].add_input_files(input_files)

        cf_daemon["scheduler_job"].build()

        utils.write_session_args(
            args,
            follower_id,
            cf_daemon["scheduler_job"]._cache_dir,
            coordinator_deadline,
            args.nfollowers,
        )

        cf_daemon["scheduler_job"].submit()
        cf_daemons["followers"].append(cf_daemon)

    return cf_daemons


def initialise_cmf_daemons(args, coordinator_deadline):
    if Path(Path.cwd(), "config", "follower_config.cfg").is_file():
        follower_config_path = Path(Path.cwd(), "config", "follower_config.cfg")
    elif Path(definitions.PACKAGE_CONFIG_DIR, "follower_config.cfg").is_file():
        follower_config_path = Path(
            definitions.PACKAGE_CONFIG_DIR, "follower_config.cfg"
        )
    else:
        raise FileNotFoundError("Could not find follower_config.cfg")

    if Path(Path.cwd(), "config", "manager_config.cfg").is_file():
        manager_config_path = Path(Path.cwd(), "config", "manager_config.cfg")
    elif Path(definitions.PACKAGE_CONFIG_DIR, "manager_config.cfg").is_file():
        manager_config_path = Path(
            definitions.PACKAGE_CONFIG_DIR, "manager_config.cfg"
        )
    else:
        raise FileNotFoundError("Could not find manager_config.cfg")

    # Archive the CondorSMC package
    utils.archive_condorsmc(
        definitions.PACKAGE_ROOT_DIR.parent,
        Path(
            definitions.SESSION_CACHE_DIR(args.session_id),
            "condorsmc",
        ),
    )

    # Archive the CondorCMF package
    utils.archive_condorsmc(
        CondorCMFDefinitions.PACKAGE_ROOT_DIR.parent,
        Path(
            definitions.SESSION_CACHE_DIR(args.session_id),
            "condorcmf",
        ),
    )

    cmf_daemons = {
        "coordinator": {
            "id": args.node_id,
        },
        "managers": [],
        "followers": [],
    }

    executable_path = Path(definitions.PACKAGE_ROOT_DIR, "bin", "daemon.bat")

    start = time()
    for i in range(args.nmanagers):
        n_followers = args.nfollowers // args.nmanagers
        n_followers_remainder = args.nfollowers % args.nmanagers
        if i < n_followers_remainder:
            n_followers += 1
    
        manager_id = f"test_manager_{str(i).zfill(4)}" if args.session_id == "test" else str(uuid.uuid4())

        cmf_manager_daemon = {
            "id": manager_id,
            "followers": [],
        }

        payload = {
            "session_id": args.session_id,
            "start_time": start,
            "session_deadline": coordinator_deadline,
            "coordinator_id": args.node_id,
            "job_id": manager_id,
        }

        cmf_manager_daemon["scheduler_job"] = SchedulerJob(
            executable=executable_path,
            requests=[("request_cpus", 1)],
            # requirements='(Arch=="X86_64") && (OpSys=="Windows")',
            requirements='(Arch=="X86_64") && (OpSys=="Windows") && (DA_CDT_HOST==TRUE)',
            # requirements='(Arch=="X86_64") && (OpSys=="LINUX") && (DA_CDT_HOST==TRUE)',
            custom_args=["+DACDTJob = true"],
            session_id=args.session_id,
            job_id=manager_id,
            payload=payload,
        )

        input_files = [
            Path(
                definitions.SESSION_CACHE_DIR(args.session_id),
                "condorsmc.zip",
            ),
            Path(
                definitions.SESSION_CACHE_DIR(args.session_id),
                "condorcmf.zip",
            ),
            PYTHON_ENV,
            manager_config_path,
            Path(cmf_manager_daemon["scheduler_job"]._cache_dir, "session_args.txt"),
        ]

        if Path(args.model_dir, f"{args.model}.py").is_file():
            model_path = Path(args.model_dir, f"{args.model}.py")
        elif Path(args.model_dir, f"{args.model}.stan").is_file():
            model_path = Path(args.model_dir, f"{args.model}.stan")
        model_data = Path(args.model_dir, f"{args.model}.json")

        input_files.append(model_path)
        if model_data.is_file():
            input_files.append(model_data)

        cmf_manager_daemon["scheduler_job"].add_input_files(input_files)

        for j in range(n_followers):
            follower_id = f"test_manager_{str(i).zfill(4)}_follower_{str(i).zfill(4)}" if args.session_id == "test" else str(uuid.uuid4())
            cmf_manager_daemon["followers"].append(follower_id)

            cmf_follower_daemon = {
                "id": follower_id,
                "manager": manager_id,
            }

            payload = {
                "session_id": args.session_id,
                "start_time": start,
                "session_deadline": coordinator_deadline,
                "coordinator_id": args.node_id,
                "manager_id": manager_id,            
                "job_id": follower_id,
            }

            cmf_follower_daemon["scheduler_job"] = SchedulerJob(
                executable=executable_path,
                requests=[("request_cpus", 1)],
                requirements='(Arch=="X86_64") && (OpSys=="Windows")',
                # requirements='(Arch=="X86_64") && (OpSys=="Windows") && (DA_CDT_HOST==TRUE)',
                # requirements='(Arch=="X86_64") && (OpSys=="LINUX") && (DA_CDT_HOST==TRUE)',
                # custom_args=["+DACDTJob = true"],
                session_id=args.session_id,
                job_id=follower_id,
                payload=payload,
            )

            input_files = [
                Path(
                    definitions.SESSION_CACHE_DIR(args.session_id),
                    "condorsmc.zip",
                ),
                Path(
                    definitions.SESSION_CACHE_DIR(args.session_id),
                    "condorcmf.zip",
                ),
                PYTHON_ENV,
                follower_config_path,
                Path(cmf_follower_daemon["scheduler_job"]._cache_dir, "session_args.txt"),
            ]

            model_path = Path(args.model_dir, f"{args.model}.py")
            model_data = Path(args.model_dir, f"{args.model}.json")

            input_files.append(model_path)
            if model_data.is_file():
                input_files.append(model_data)

            cmf_follower_daemon["scheduler_job"].add_input_files(input_files)

            cmf_daemons["followers"].append(cmf_follower_daemon)

        cmf_daemons["managers"].append(cmf_manager_daemon)

    for cmf_manager_daemon in cmf_daemons["managers"]:
        cmf_manager_daemon["scheduler_job"].payload["followers"] = cmf_manager_daemon["followers"]
        cmf_manager_daemon["scheduler_job"].build()

        utils.write_session_args(
            args,
            follower_id,
            cmf_manager_daemon["scheduler_job"]._cache_dir,
            coordinator_deadline,
            len(cmf_manager_daemon["followers"]),
            "manager",
        )

        cmf_manager_daemon["scheduler_job"].submit()

    for cmf_follower_daemon in cmf_daemons["followers"]:
        cmf_follower_daemon["scheduler_job"].payload["manager"] = cmf_follower_daemon["manager"]
        cmf_follower_daemon["scheduler_job"].build()

        utils.write_session_args(
            args,
            follower_id,
            cmf_follower_daemon["scheduler_job"]._cache_dir,
            coordinator_deadline,
            args.nfollowers,
            "follower",
        )

        cmf_follower_daemon["scheduler_job"].submit()

    return cmf_daemons


def load_network(args, coordinator_deadline, struct_path):
    """
    Read network structure from a yaml file and load the daemons into the network.
    """

    with open(struct_path, "r") as f:
        session_daemons = yaml.safe_load(f)

    specified_followers = "followers" in session_daemons.keys()

    if "managers" in session_daemons.keys() and args.nmanagers != len(session_daemons["managers"]):
        raise ValueError(f"Number of managers in network structure does not match number of managers in session.")
    
    if args.nmanagers == 0 and "followers" not in session_daemons.keys():
        raise ValueError(f"Must specify follower network structure when running with no managers.")
    elif "followers" in session_daemons.keys() and args.nfollowers != len(session_daemons["followers"]):
        raise ValueError(f"Number of followers in network structure does not match number of followers in session.")

    if Path(Path.cwd(), "config", "follower_config.cfg").is_file():
        follower_config_path = Path(Path.cwd(), "config", "follower_config.cfg")
    elif Path(definitions.PACKAGE_CONFIG_DIR, "follower_config.cfg").is_file():
        follower_config_path = Path(
            definitions.PACKAGE_CONFIG_DIR, "follower_config.cfg"
        )
    else:
        raise FileNotFoundError("Could not find follower_config.cfg")

    # Archive the CondorSMC package
    utils.archive_condorsmc(
        definitions.PACKAGE_ROOT_DIR.parent,
        Path(
            definitions.SESSION_CACHE_DIR(args.session_id),
            "condorsmc",
        ),
    )

    # Archive the CondorCMF package
    utils.archive_condorsmc(
        CondorCMFDefinitions.PACKAGE_ROOT_DIR.parent,
        Path(
            definitions.SESSION_CACHE_DIR(args.session_id),
            "condorcmf",
        ),
    )

    executable_path = Path(definitions.PACKAGE_ROOT_DIR, "bin", "daemon.bat")

    start = time()
    # Spawn CMF network daemons from network structure
    if "managers" in session_daemons.keys():
        if Path(Path.cwd(), "config", "manager_config.cfg").is_file():
            manager_config_path = Path(Path.cwd(), "config", "manager_config.cfg")
        elif Path(definitions.PACKAGE_CONFIG_DIR, "manager_config.cfg").is_file():
            manager_config_path = Path(
                definitions.PACKAGE_CONFIG_DIR, "manager_config.cfg"
            )
        else:
            raise FileNotFoundError("Could not find manager_config.cfg")

        if not "followers" in session_daemons.keys():
            session_daemons["followers"] = []

        for i, cmf_manager_daemon in enumerate(session_daemons["managers"]):
            payload = {
                "session_id": args.session_id,
                "start_time": start,
                "session_deadline": coordinator_deadline,
                "coordinator_id": args.node_id,
                "job_id": cmf_manager_daemon["id"],
            }

            if "followers" in cmf_manager_daemon.keys():
                payload["followers"] = cmf_manager_daemon["followers"]
            else:
                cmf_manager_daemon["followers"] = []

            cmf_manager_daemon["scheduler_job"] = SchedulerJob(
                executable=executable_path,
                requests=[("request_cpus", 1)],
                requirements= cmf_manager_daemon["requirements"] if "requirements" in cmf_manager_daemon else '(Arch=="X86_64") && (OpSys=="Windows") && (DA_CDT_HOST==TRUE)',
                custom_args= cmf_manager_daemon["custom_args"] if "custom_args" in cmf_manager_daemon else [],
                session_id=args.session_id,
                job_id=cmf_manager_daemon["id"],
                payload=payload,
            )

            input_files = [
                Path(
                    definitions.SESSION_CACHE_DIR(args.session_id),
                    "condorsmc.zip",
                ),
                Path(
                    definitions.SESSION_CACHE_DIR(args.session_id),
                    "condorcmf.zip",
                ),
                PYTHON_ENV,
                manager_config_path,
                Path(cmf_manager_daemon["scheduler_job"]._cache_dir, "session_args.txt"),
            ]

            if Path(args.model_dir, f"{args.model}.py").is_file():
                model_path = Path(args.model_dir, f"{args.model}.py")
            elif Path(args.model_dir, f"{args.model}.stan").is_file():
                model_path = Path(args.model_dir, f"{args.model}.stan")
            model_data = Path(args.model_dir, f"{args.model}.json")

            input_files.append(model_path)
            if model_data.is_file():
                input_files.append(model_data)

            cmf_manager_daemon["scheduler_job"].add_input_files(input_files)

            # Followers not specified in network structure, so we generate args.nfollowers followers and 
            # distribute them evenly across the managers
            if args.nmanagers > 0 and not specified_followers:
                n_followers = args.nfollowers // args.nmanagers
                n_followers_remainder = args.nfollowers % args.nmanagers
                if i < n_followers_remainder:
                    n_followers += 1

                for j in range(n_followers):
                    follower_id = f"{cmf_manager_daemon['id']}_follower_{j}" if args.session_id == "test" else str(uuid.uuid4())
                    cmf_manager_daemon["followers"].append(follower_id)

                    cmf_follower_daemon = {
                        "id": follower_id,
                        "manager": cmf_manager_daemon["id"],
                    }

                    payload = {
                        "session_id": args.session_id,
                        "start_time": start,
                        "session_deadline": coordinator_deadline,
                        "coordinator_id": args.node_id,
                        "job_id": follower_id,
                        "manager": cmf_manager_daemon["id"],
                    }

                    cmf_follower_daemon["scheduler_job"] = SchedulerJob(
                        executable=executable_path,
                        requests=[("request_cpus", 1)],
                        requirements= cmf_follower_daemon["requirements"] if "requirements" in cmf_follower_daemon else '(Arch=="X86_64") && (OpSys=="Windows")',
                        custom_args= cmf_follower_daemon["custom_args"] if "custom_args" in cmf_follower_daemon else [],
                        session_id=args.session_id,
                        job_id=follower_id,
                        payload=payload,
                    )

                    input_files = [
                        Path(
                            definitions.SESSION_CACHE_DIR(args.session_id),
                            "condorsmc.zip",
                        ),
                        Path(
                            definitions.SESSION_CACHE_DIR(args.session_id),
                            "condorcmf.zip",
                        ),
                        PYTHON_ENV,
                        follower_config_path,
                        Path(cmf_follower_daemon["scheduler_job"]._cache_dir, "session_args.txt"),
                    ]

                    if Path(args.model_dir, f"{args.model}.py").is_file():
                        model_path = Path(args.model_dir, f"{args.model}.py")
                    elif Path(args.model_dir, f"{args.model}.stan").is_file():
                        model_path = Path(args.model_dir, f"{args.model}.stan")
                    model_data = Path(args.model_dir, f"{args.model}.json")

                    input_files.append(model_path)
                    if model_data.is_file():
                        input_files.append(model_data)

                    cmf_follower_daemon["scheduler_job"].add_input_files(input_files)

                    session_daemons["followers"].append(cmf_follower_daemon)
            else:
                cmf_manager_daemon["scheduler_job"].build()

                utils.write_session_args(
                    args,
                    cmf_manager_daemon["id"],
                    cmf_manager_daemon["scheduler_job"]._cache_dir,
                    coordinator_deadline,
                    len(cmf_manager_daemon["followers"]),
                    "manager",
                )

                cmf_manager_daemon["scheduler_job"].submit()

    if args.nmanagers > 0 and not specified_followers:
        for cmf_manager_daemon in session_daemons["managers"]:
            cmf_manager_daemon["scheduler_job"].payload["followers"] = cmf_manager_daemon["followers"]
            cmf_manager_daemon["scheduler_job"].build()

            utils.write_session_args(
                args,
                cmf_manager_daemon["id"],
                cmf_manager_daemon["scheduler_job"]._cache_dir,
                coordinator_deadline,
                len(cmf_manager_daemon["followers"]),
                "manager",
            )

            cmf_manager_daemon["scheduler_job"].submit()

        for cmf_follower_daemon in session_daemons["followers"]:
            cmf_follower_daemon["scheduler_job"].payload["manager"] = cmf_follower_daemon["manager"]
            cmf_follower_daemon["scheduler_job"].build()

            utils.write_session_args(
                args,
                cmf_follower_daemon["id"],
                cmf_follower_daemon["scheduler_job"]._cache_dir,
                coordinator_deadline,
                args.nfollowers,
                "follower",
            )

            cmf_follower_daemon["scheduler_job"].submit()
    else:
        for cmf_follower_daemon in session_daemons["followers"]:
            payload = {
                "session_id": args.session_id,
                "start_time": start,
                "session_deadline": coordinator_deadline,
                "coordinator_id": args.node_id,
                "job_id": cmf_follower_daemon["id"],
            }

            if "manager" in cmf_follower_daemon:
                payload["manager"] = cmf_follower_daemon["manager"]

            cmf_follower_daemon["scheduler_job"] = SchedulerJob(
                executable=executable_path,
                requests=[("request_cpus", 1)],
                requirements= cmf_follower_daemon["requirements"] if "requirements" in cmf_follower_daemon else '(Arch=="X86_64") && (OpSys=="Windows")',
                custom_args= cmf_follower_daemon["custom_args"] if "custom_args" in cmf_follower_daemon else [],
                session_id=args.session_id,
                job_id=cmf_follower_daemon["id"],
                payload=payload,
            )

            input_files = [
                Path(
                    definitions.SESSION_CACHE_DIR(args.session_id),
                    "condorsmc.zip",
                ),
                Path(
                    definitions.SESSION_CACHE_DIR(args.session_id),
                    "condorcmf.zip",
                ),
                PYTHON_ENV,
                follower_config_path,
                Path(cmf_follower_daemon["scheduler_job"]._cache_dir, "session_args.txt"),
            ]

            if Path(args.model_dir, f"{args.model}.py").is_file():
                model_path = Path(args.model_dir, f"{args.model}.py")
            elif Path(args.model_dir, f"{args.model}.stan").is_file():
                model_path = Path(args.model_dir, f"{args.model}.stan")
            model_data = Path(args.model_dir, f"{args.model}.json")

            input_files.append(model_path)
            if model_data.is_file():
                input_files.append(model_data)

            cmf_follower_daemon["scheduler_job"].add_input_files(input_files)

            cmf_follower_daemon["scheduler_job"].build()

            utils.write_session_args(
                args,
                cmf_follower_daemon["id"],
                cmf_follower_daemon["scheduler_job"]._cache_dir,
                coordinator_deadline,
                args.nfollowers,
                "follower",
            )

            cmf_follower_daemon["scheduler_job"].submit()

    return session_daemons
