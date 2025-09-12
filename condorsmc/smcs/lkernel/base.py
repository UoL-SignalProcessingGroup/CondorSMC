import numpy.typing as npt

from abc import ABC, abstractmethod

from ..state import SMCState


class LKernelBase(ABC):
    @abstractmethod
    def calculate_rw(self, 
                     smc_state : SMCState
                     ) -> SMCState:
        pass
    
    @abstractmethod
    def calculate_hmc(self, 
                     smc_state : SMCState
                     ) -> SMCState:
        pass
