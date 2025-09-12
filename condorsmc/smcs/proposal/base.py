import autograd.numpy as np
import numpy.typing as npt
from numpy.random import Generator

from abc import ABC, abstractmethod

from ..state import SMCState
from ..target import TargetBase

class ProposalBase(ABC):
    def __init__(self) -> None:
        self.target : TargetBase = None
        self.rng : Generator = None

    @abstractmethod
    def propose(self,
                smc_state : SMCState
                ) -> SMCState:
        pass

    @abstractmethod
    def logpdf(self,
               smc_state : SMCState
               ) -> npt.NDArray[np.float_]:
        pass
