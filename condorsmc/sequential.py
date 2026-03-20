# condorsmc/sequential.py
from __future__ import annotations

import logging
import autograd.numpy as np
from time import time

from . import definitions, utils, smcs


def mse(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return np.mean((a - b) ** 2)


def main(args):
    log = logging.getLogger(__name__)

    target = utils.load_target(args.model_dir, args.model)

    executor = smcs.executor.SequentialExecutor()
    rng = smcs.set_seed(args.seed, rank=0)
    dim = getattr(target, "constrained_dim", target.dim)

    forward_kernel = smcs.create_proposal(
        target=target,
        proposal_args={"step_size": 0.5, "rng": rng},
        accept_reject=False,
    )
    lkernel = smcs.create_lkernel(forward_kernel, executor=executor)
    recycling = smcs.create_recycling(args.recycling, executor=executor)

    initialiser = smcs.initialise.GaussianInitialiser(target, forward_kernel, executor, rng)
    adaption = smcs.adaption.ParallelWindowedAdaption(target, forward_kernel, lkernel, recycling, executor, rng)

    adaption_stats = smcs.SMCStatistics(args.nsamples, dim, executor)
    initial_state = initialiser.init(args.nsamples)

    t0 = time()
    smc_state, forward_kernel, lkernel = adaption.adapt(initial_state, adaption_stats)
    adaption_time = time() - t0

    sampler = smcs.SMCSampler(
        target=target,
        forward_kernel=forward_kernel,
        lkernel=lkernel,
        recycling=recycling,
        executor=executor,
    )

    smc_stats = smcs.SMCStatistics(args.nsamples, dim, executor)
    smc_state = sampler.sample(
        smc_state,
        smc_stats,
        args.niters,
        record_states=False,
    )

    mean_last = np.asarray(smc_stats.mean_estimate[-1])
    var_last = np.asarray(smc_stats.variance_estimate[-1])

    true_mean = getattr(args, "true_mean", None)
    true_variance = getattr(args, "true_variance", None)
    mse_mean = float(mse(mean_last, np.asarray(true_mean))) if true_mean is not None else np.nan
    mse_std  = float(mse(np.sqrt(var_last), np.sqrt(np.asarray(true_variance)))) if true_variance is not None else np.nan

    print(f"Adaption time: {adaption_time:.5f}s")
    if hasattr(smc_stats, "run_time"):
        print(f"Sampling time: {smc_stats.run_time:.5f}s")
    print(f"Mean: {mean_last}, MSE(mean): {mse_mean}")
    print(f"Std:  {np.sqrt(var_last)}, MSE(std): {mse_std}")
    print("-" * 50)

    outdir = definitions.SESSION_OUTPUT_DIR(args.session_id)
    outdir.mkdir(parents=True, exist_ok=True)

    names = getattr(target, "param_names", None)
    if isinstance(names, (list, tuple)) and len(names) == mean_last.size:
        labels = list(names)
    else:
        if names is not None and hasattr(names, "__len__") and len(names) != mean_last.size:
            log.warning(
                "param_names length (%s) != parameter dimension (%s); falling back to indices.",
                len(names), mean_last.size,
            )
        labels = [str(i) for i in range(mean_last.size)]

    mean_rows = np.column_stack([np.array(labels, dtype=object), mean_last.astype(float)])
    np.savetxt(
        outdir / "mean_estimate.csv",
        mean_rows,
        fmt=["%s", "%.18e"],
        delimiter=",",
        header="param,mean",
        comments="",
    )

    var_rows = np.column_stack([np.array(labels, dtype=object), var_last.astype(float)])
    np.savetxt(
        outdir / "variance_estimate.csv",
        var_rows,
        fmt=["%s", "%.18e"],
        delimiter=",",
        header="param,variance",
        comments="",
    )

    with open(outdir / "summary.csv", "w") as f:
        f.write("metric,value\n")
        f.write(f"mse_mean,{mse_mean}\n")
        f.write(f"mse_std,{mse_std}\n")
        if hasattr(smc_stats, "run_time"):
            f.write(f"sampling_time,{smc_stats.run_time}\n")
        f.write(f"adaption_time,{adaption_time}\n")

    print(f"Saved mean/variance CSVs and summary to {outdir}")
