import datetime
from pathlib import Path

import numpy as np

from . import definitions


def write_session_info(args, output_dir, write_output=True):
    """Write the session information to the output directory.

    Parameters
    ----------
    args : argparse.Namespace
        The command-line arguments.
    output_dir : str
        The output directory.
    """
    print("------------ CondorSMC ------------")
    print(f"Session ID: {args.session_id}")
    print(f"Node ID: {args.node_id}")
    print(f"Node Role: {args.role}")
    print(f"Initialised at: {datetime.datetime.now()}")
    print(f"Using seed: {args.seed}")
    print("-----------------------------------")
    print(f"Sampling Mode: {args.mode}")
    if args.mode == "distributed":
        # Convert runtime to hours, minutes, seconds
        hours, remainder = divmod(args.coordinator_runtime, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"Coordinator runtime: {hours:.0f}h {minutes:.0f}m {seconds:.0f}s")
        if args.nmanagers > 1:
            print(f"Number of managers: {args.nmanagers}")
            hours, remainder = divmod(args.manager_runtime, 3600)
            minutes, seconds = divmod(remainder, 60)
            print(f"Manager runtime: {hours:.0f}h {minutes:.0f}m {seconds:.0f}s")
        print(f"Number of followers: {args.nfollowers}")
        hours, remainder = divmod(args.follower_runtime, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"Follower runtime: {hours:.0f}h {minutes:.0f}m {seconds:.0f}s")
        print(f"Follower max sampling iters: {args.niters}")
    print(f"Sampling Model: {args.model}")
    print("-------SMCS Hyperparameters--------")
    print(f"Number of Samples: {args.nsamples}")
    print(f"Number of Iterations: {args.niters}")
    print(f"Proposal: {args.proposal}")
    if args.proposal == "hmc" or args.proposal == "nuts":
        print(f"Integrator: {args.integrator}")
        print(f"Step Size: {args.step_size}")
        print(f"Number of Steps: {args.hmc_steps}")
    print(f"L Kernel: {args.lkernel}")
    print(f"Recycling: {args.recycling}")
    print(f"Resampling: {args.resampling}")
    print("-----------------------------------")

    if write_output:
        with open(Path(output_dir, "session_info.txt"), "w") as f:
            f.write("------------ CondorSMC ------------\n")
            f.write(f"Session ID: {args.session_id}\n")
            f.write(f"Node ID: {args.node_id}\n")
            f.write(f"Seed: {args.seed}\n")
            f.write("-----------------------------------\n")
            f.write(f"Sampling Mode: {args.mode}\n")
            if args.mode == "distributed":
                hours, remainder = divmod(args.coordinator_runtime, 3600)
                minutes, seconds = divmod(remainder, 60)
                f.write(f"Coordinator runtime: {hours:.0f}h {minutes:.0f}m {seconds:.0f}s\n")
                if args.nmanagers > 1:
                    f.write(f"Number of managers: {args.nmanagers}\n")
                    hours, remainder = divmod(args.manager_runtime, 3600)
                    minutes, seconds = divmod(remainder, 60)
                    f.write(f"Manager runtime: {hours:.0f}h {minutes:.0f}m {seconds:.0f}s\n")
                f.write(f"Number of followers: {args.nfollowers}\n")
                hours, remainder = divmod(args.follower_runtime, 3600)
                minutes, seconds = divmod(remainder, 60)
                f.write(f"Follower runtime: {hours:.0f}h {minutes:.0f}m {seconds:.0f}s\n")
                f.write(f"Follower max sampling iters:: {args.niters}\n")
            f.write(f"Sampling Model: {args.model}\n")
            f.write("-------SMCS Hyperparameters--------\n")
            f.write(f"Number of Samples: {args.nsamples}\n")
            f.write(f"Number of Iterations: {args.niters}\n")
            f.write(f"Proposal: {args.proposal}\n")
            if args.proposal == "hmc" or args.proposal == "nuts":
                f.write(f"Integrator: {args.integrator}\n")
                f.write(f"Step Size: {args.step_size}\n")
                f.write(f"Number of Steps: {args.hmc_steps}\n")
            f.write(f"L Kernel: {args.lkernel}\n")
            f.write(f"Recycling: {args.recycling}\n")
            f.write(f"Resampling: {args.resampling}\n")
            f.write("-----------------------------------\n")


def write_error_report(args, exception, traceback, outputdir):
    """Write an error report to the output directory.

    Parameters
    ----------
    args : argparse.Namespace
        The command-line arguments.
    exception : Exception
        The exception that was raised.
    outputdir : str
        The output directory.
    """
    with open(Path(outputdir, "error.txt"), "w") as f:
        f.write("------------ CondorSMC ------------\n")
        f.write(f"Session ID: {args.session_id}\n")
        f.write(f"Node ID: {args.node_id}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write("-----------------------------------\n")
        f.write(f"Sampling Mode: {args.mode}\n")
        f.write(f"Sampling Model: {args.model}\n")
        f.write("-------SMCS Hyperparameters--------\n")
        f.write(f"Number of Samples: {args.nsamples}\n")
        f.write(f"Number of Iterations: {args.niters}\n")
        f.write(f"Proposal: {args.proposal}\n")
        if args.proposal == "hmc" or args.proposal == "nuts":
            f.write(f"Integrator: {args.integrator}\n")
            f.write(f"Step Size: {args.step_size}\n")
            f.write(f"Number of Steps: {args.hmc_steps}\n")
        f.write(f"L Kernel: {args.lkernel}\n")
        f.write(f"Recycling: {args.recycling}\n")
        f.write("-----------------------------------\n")
        f.write(f"Exception:\n")
        f.write(f"{exception}\n")
        f.write("-----------------------------------\n")
        f.write(f"Traceback:\n")
        f.write(f"{traceback}\n")
        f.write("-----------------------------------\n")


def write_results(
    args, outputdir, mean, variance, recycled_mean=None, recycled_variance=None
):
    np.savetxt(Path(outputdir, "mean_estimate.txt"), mean, delimiter=",")
    np.savetxt(Path(outputdir, "variance_estimate.txt"), variance, delimiter=",")
    if recycled_mean is not None and recycled_variance is not None:
        np.savetxt(Path(outputdir, "recycled_mean_estimate.txt"), recycled_mean, delimiter=",")
        np.savetxt(Path(outputdir, "recycled_variance_estimate.txt"), recycled_variance, delimiter=",")
