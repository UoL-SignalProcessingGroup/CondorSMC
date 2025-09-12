import autograd.numpy as np

from abc import ABC, abstractmethod

from ..state import SMCState
from ..executor.sequential import SequentialExecutor


class InitialiserBase(ABC):
    def __init__(self, target, proposal, executor=SequentialExecutor(), rng=np.random.default_rng()):
        self.target = target
        self.proposal = proposal
        self.executor = executor
        self.rng = rng

    @abstractmethod
    def init(self, num_particles) -> SMCState:
        pass
