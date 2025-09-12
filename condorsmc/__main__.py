from __future__ import annotations

import argparse
import logging
import shutil
import sys
import traceback
import uuid
from pathlib import Path

from . import coordinator, definitions, follower, manager, sequential, writer
from condorcmf.scheduler import utils as SchedulerUtils

VALID_PROPOSALS = ("nuts")
VALID_LKERNELS = ("reverse_proposal")
VALID_RECYCLING = ("none", "ess")
VALID_RESAMPLING = ("centralised", "centralised_ews", "centralised_nodes")
VALID_ROLES = ("coordinator", "manager", "follower")


def build_common_parser(p: argparse.ArgumentParser) -> None:
    # identity / sessioning
    p.add_argument("--session-id", type=str, default=str(uuid.uuid4()),
                   help="Unique session ID (default: random UUID).")
    p.add_argument("--node-id", type=str, default=str(uuid.uuid4()),
                   help="Unique node ID (default: random UUID).")

    # model
    p.add_argument("--model-dir", type=str, default=str(Path.cwd()),
                   help="Path to the model directory.")
    p.add_argument("--model", type=str, required=True,
                   help="Name of the model to run (required).")

    # SMC core
    p.add_argument("--nsamples", type=int, default=100,
                   help="Number of samples to generate.")
    p.add_argument("--recycling", choices=VALID_RECYCLING, default="none",
                   help="Recycling scheme (default: none).")

    # misc
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    p.add_argument("--verbose", action="store_true", help="Verbose logging.")
    p.add_argument("--nowait", action="store_true",
                   help="Do not wait for all nodes to finish before exiting.")
    p.add_argument("--network-structure", type=str, default=None,
                   help="YAML file describing the network structure.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch CondorSMC.")

    subparsers = parser.add_subparsers(dest="mode", required=False)

    # sequential mode (DEFAULT)
    seq = subparsers.add_parser("sequential", help="Run in single-process sequential mode.")
    build_common_parser(seq)
    seq.add_argument("--niters", type=int, required=True,
                     help="Number of iterations to run.")

    # distributed mode
    dist = subparsers.add_parser("distributed", help="Run in distributed mode.")
    build_common_parser(dist)
    dist.add_argument("--role", choices=VALID_ROLES, required=True,
                      help="Role of this node.")
    dist.add_argument("--nmanagers", type=int, default=0,
                      help="Number of manager nodes to use.")
    dist.add_argument("--nfollowers", type=int, default=1,
                      help="Number of follower nodes to use.")
    dist.add_argument("--coordinator-runtime", type=float, default=30 * 60,
                      help="Coordinator runtime (seconds).")
    dist.add_argument("--manager-runtime", type=float, default=10 * 60,
                      help="Manager runtime (seconds).")
    dist.add_argument("--follower-runtime", type=float, default=30,
                      help="Follower runtime (seconds; max sampling time).")

    return parser


def parse_args_defaulting_to_sequential(parser: argparse.ArgumentParser) -> argparse.Namespace:
    argv = sys.argv[1:]
    if not argv or argv[0] not in ("sequential", "distributed"):
        argv = ["sequential"] + argv
    return parser.parse_args(argv)


def prepare_output_dir(session_id: str, verbose: bool) -> None:
    outdir = definitions.SESSION_OUTPUT_DIR(session_id)
    if session_id == "test" and outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if verbose:
        logging.basicConfig(level=logging.INFO)

    logging.info("Starting CondorSMC session %s", session_id)


def run_sequential(args: argparse.Namespace) -> None:
    # writer.write_session_info(args, f"{definitions.SESSION_OUTPUT_DIR(args.session_id)}")
    sequential.main(args=args)


def run_distributed(args: argparse.Namespace) -> None:
    if args.role == "coordinator":
        if args.session_id == "test":
            args.node_id = "test_coordinator"
        writer.write_session_info(args, f"{definitions.SESSION_OUTPUT_DIR(args.session_id)}")
        try:
            coordinator.main(args=args)
        except KeyboardInterrupt:
            print("Keyboard interrupt detected, exiting...")
            SchedulerUtils.condor_remove("--all")
        except Exception as e:
            print("------- EXCEPTION DETECTED -------")
            print("Please raise an issue on GitHub with the full error report which can be found in:")
            print(f"{definitions.SESSION_OUTPUT_DIR(args.session_id)}/error.txt")
            print("https://github.com/mjcarter95/CondorSMC/issues")
            writer.write_error_report(
                args,
                e,
                traceback.format_exc(),
                f"{definitions.SESSION_OUTPUT_DIR(args.session_id)}",
            )
    elif args.role == "manager":
        writer.write_session_info(
            args, f"{definitions.SESSION_OUTPUT_DIR(args.session_id)}", write_output=False
        )
        manager.main(args=args)
    elif args.role == "follower":
        writer.write_session_info(
            args, f"{definitions.SESSION_OUTPUT_DIR(args.session_id)}", write_output=False
        )
        follower.main(args=args)
    else:
        raise ValueError(f"Unknown role {args.role!r}")


def main() -> None:
    parser = build_parser()
    args = parse_args_defaulting_to_sequential(parser)

    prepare_output_dir(args.session_id, args.verbose)

    mode = args.mode or "sequential"

    if mode == "sequential":
        run_sequential(args)
    elif mode == "distributed":
        run_distributed(args)
    else:
        raise ValueError(f"Unknown mode {mode!r}")

    logging.info("Ending CondorSMC session %s", args.session_id)


if __name__ == "__main__":
    main()
