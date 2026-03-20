from __future__ import annotations

import autograd.numpy as np
from time import time
from typing import Any, Union, Optional, Tuple

from numpy.typing import ArrayLike
from scipy.stats import multivariate_normal

from .executor.sequential import SequentialExecutor
from .lkernel import LKernelBase, ForwardLKernel
from .proposal import ProposalBase, NUTSProposal
from .proposal.nuts import NUTSProposalAcceptReject
from .recycling import (
    RecyclingBase, NoRecycling, ESSRecycling
)



def set_seed(
    seed: Union[None, int],
    rank: int = 0,
):
    if seed:
        return np.random.default_rng(seed)
    from time import time as _now
    return np.random.default_rng(int(_now()) + rank)


def create_proposal(
    target: TargetBase,
    proposal_args: Optional[dict[str, Any]],
    accept_reject: bool = False,
) -> ProposalBase:

    momentum_proposal = multivariate_normal(np.zeros(target.dim), np.eye(target.dim))

    # nuts
    proposal_args = {"step_size": 0.5} if proposal_args is None else proposal_args
    klass = NUTSProposalAcceptReject if accept_reject else NUTSProposal
    return klass(target=target, momentum_proposal=momentum_proposal, **proposal_args)


def create_lkernel(
    forward_kernel: ProposalBase,
    executor: Optional[SequentialExecutor] = None,
) -> LKernelBase:

    return ForwardLKernel(forward_kernel=forward_kernel)


def create_recycling(
    recycling_scheme: str,
    executor: Optional[SequentialExecutor] = None,
) -> RecyclingBase:
    available = ("none", "ess")
    if recycling_scheme not in available:
        raise ValueError(f"Invalid recycling scheme '{recycling_scheme}'. Available: {', '.join(available)}.")

    if recycling_scheme == "ess":
        return ESSRecycling(executor=executor or SequentialExecutor())

    return NoRecycling()
