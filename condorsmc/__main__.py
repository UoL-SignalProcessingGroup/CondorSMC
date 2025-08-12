import argparse
import datetime
import logging
import shutil
import traceback
import uuid
from pathlib import Path

from . import coordinator, definitions, follower, manager, sequential, writer
from condorcmf.scheduler import utils as SchedulerUtils


def none_or_int(value):
    if value.lower() == "none":
        return None
    return int(value)


def none_or_str(value):
    if value.lower() == "none":
        return None
    return str(value)


parser = argparse.ArgumentParser(description="Launch CondorSMC.")
parser.add_argument(
    "--session-id",
    type=str,
    default=str(uuid.uuid4()),
    help="Role of this node (coordinator, manager, worker, or sequential).",
)
parser.add_argument(
    "--node-id",
    type=str,
    default=str(uuid.uuid4()),
    help="Role of this node (coordinator, manager, worker, or sequential).",
)
parser.add_argument(
    "--mode",
    type=str,
    default="sequential",
    help="Sampling mode (sequential, distributed).",
)
parser.add_argument(
    "--role",
    type=str,
    default="coordinator",
    help="Role of this node (coordinator, manager, follower).",
)

# condorsmc arguments
parser.add_argument(
    "--coordinator-runtime",
    type=float,
    # default=400,
    default=30*60,
    help="Runtime of the coordinator node in seconds.",
)
parser.add_argument(
    "--nmanagers", type=int, default=0, help="Number of manager nodes to use."
)
parser.add_argument(
    "--manager-runtime",
    type=float,
    # default=40,
    default=10*60,
    help="Runtime of the manager nodes in seconds.",
)
parser.add_argument(
    "--nfollowers", type=int, default=1, help="Number of follower nodes to use."
)
parser.add_argument(
    "--follower-runtime",
    type=float,
    # default=10,
    default=30,
    help="Runtime of the follower nodes in seconds (max sampling time).",
)

# model arguments
parser.add_argument(
    "--model-dir", type=str, default=str(Path.cwd()), help="Path to the model file."
)
parser.add_argument("--model", type=str, default=None, help="Name of the model.")

# smcs arguments
parser.add_argument(
    "--nsamples", type=int, default=100, help="Number of samples to generate."
)
parser.add_argument(
    "--niters",
    type=none_or_int,
    default=None,
    help="Number of iterations to run, in distributed mode this is the maximum number of sampling iterations a follower will complete.",
)
parser.add_argument(
    "--proposal", type=str, default="rw", help="Type of proposal to use (rw, hmc)."
)
parser.add_argument(
    "--lkernel",
    type=none_or_str,
    default=None,
    help="Type of L kernel to use (none, pseudo).",
)
parser.add_argument(
    "--recycling",
    type=none_or_str,
    default=None,
    help="Type of recycling to use (none, ess).",
)
parser.add_argument(
    "--integrator",
    type=str,
    default="leapfrog",
    help="Type of integrator to use (leapfrog).",
)
parser.add_argument(
    "--resampling",
    type=none_or_str,
    default=None,
    help="Type of resampling to use (none, centralised, centralised_ews, centralised_nodes).",
)

# smcs hmc arguments
parser.add_argument("--step-size", type=float, default=0.1, help="Step size for HMC.")
parser.add_argument(
    "--hmc-steps", type=int, default=10, help="Number of steps for HMC."
)

# miscellaneous
parser.add_argument("--seed", type=int, default=0, help="Random seed to use.")
parser.add_argument("--verbose", action="store_true", help="Print verbose output.")
parser.add_argument(
    "--nowait",
    action="store_true",
    help="Do not wait for all nodes to finish before exiting.",
)
parser.add_argument("--network_structure", type=none_or_str, default=None, help="A yaml file describing the network structure.")


def validate_args(args):
    valid_modes = ["sequential", "distributed"]
    valid_roles = ["coordinator", "manager", "follower"]
    valid_proposals = ["rw", "hmc", "nuts"]
    valid_lkernels = [None, "pseudo", "gauss"]
    valid_recycling = [None, "ess"]
    valid_integrators = ["leapfrog"]
    valid_resampling = [None, "centralised", "centralised_ews", "centralised_nodes"]

    if args.model is None:
        raise ValueError("Must specify a model to run.")

    if args.mode not in valid_modes:
        raise ValueError(f"Invalid mode {args.mode}, must be one of {valid_modes}")

    if args.mode == "sequential" and args.niters is None:
        raise ValueError("Must specify number of iterations to run in sequential mode.")

    if args.role not in valid_roles:
        raise ValueError(f"Invalid role {args.role}, must be one of {valid_roles}")

    if args.proposal not in valid_proposals:
        raise ValueError(
            f"Invalid proposal {args.proposal}, must be one of {valid_proposals}"
        )

    if args.lkernel not in valid_lkernels:
        raise ValueError(
            f"Invalid lkernel {args.lkernel}, must be one of {valid_lkernels}"
        )

    if args.recycling not in valid_recycling:
        raise ValueError(
            f"Invalid recycling {args.recycling}, must be one of {valid_recycling}"
        )

    if args.integrator not in valid_integrators:
        raise ValueError(
            f"Invalid integrator {args.integrator}, must be one of {valid_integrators}"
        )

    if args.resampling not in valid_resampling:
        raise ValueError(
            f"Invalid resampling {args.resampling}, must be one of {valid_resampling}"
        )


if __name__ == "__main__":
    args = parser.parse_args()

    validate_args(args)

    if (
        args.session_id == "test"
        and definitions.SESSION_OUTPUT_DIR(args.session_id).exists()
    ):
        shutil.rmtree(definitions.SESSION_OUTPUT_DIR(args.session_id))
    definitions.SESSION_OUTPUT_DIR(args.session_id).mkdir(parents=True, exist_ok=True)

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    logging.info(f"Starting CondorSMC session {args.session_id}")

    if args.mode == "sequential":
        writer.write_session_info(
            args, f"{definitions.SESSION_OUTPUT_DIR(args.session_id)}"
        )
        sequential.main(args=args)
    elif args.role == "coordinator":
        if args.session_id == "test":
            args.node_id = "test_coordinator"
        writer.write_session_info(
            args, f"{definitions.SESSION_OUTPUT_DIR(args.session_id)}"
        )
        try:
            coordinator.main(args=args)
        except KeyboardInterrupt:
            print("Keyboard interrupt detected, exiting...")
            SchedulerUtils.condor_remove("--all")
        except Exception as e:
            print(f"------- EXCEPTION DETECTED -------")
            print(
                f"Please raise an issue on GitHub with the full error report which can be found in:"
            )
            print(f"{definitions.SESSION_OUTPUT_DIR(args.session_id)}/error.txt")
            print(f"https://github.com/mjcarter95/CondorSMC/issues")
            writer.write_error_report(
                args,
                e,
                traceback.format_exc(),
                f"{definitions.SESSION_OUTPUT_DIR(args.session_id)}",
            )
            # SchedulerUtils.condor_remove("--all")
    elif args.role == "manager":
        writer.write_session_info(
            args,
            f"{definitions.SESSION_OUTPUT_DIR(args.session_id)}",
            write_output=False,
        )
        manager.main(args=args)
    elif args.role == "follower":
        writer.write_session_info(
            args,
            f"{definitions.SESSION_OUTPUT_DIR(args.session_id)}",
            write_output=False,
        )
        follower.main(args=args)
    else:
        raise ValueError(f"Unknown role {args.role}")

    logging.info(f"Ending CondorSMC session {args.session_id}")
