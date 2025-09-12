import numpy as np
import numpy.typing as npt


class TargetBase():
    def __init__(self):
        self.dim : int = None

    def logpdf(self, 
               x : npt.NDArray[np.float64], 
               phi : int = 1.0,
               ) -> npt.NDArray[np.float64]:
        """
        Parameters
        ----------
        x : np.NDArray
            x has shape (number of samples, target dimension)

        phi : int
            controls tempering sequence

        Return
        ------
        np.NDArray
            log probability distribution funciton values for x, has shape (number of samples,)
        """
        return 
    
    def logpdfgrad(self, 
                   x : npt.NDArray[np.float64], 
                   phi : int = 1.0
                   ) -> npt.NDArray[np.float64]:
        """
        Parameters
        ----------
        x : np.NDArray
            x has shape (number of samples, target dimension)

        phi : int
            controls tempering sequence

        Return
        ------
        np.NDArray
            log probability distribution funciton gradient values for x, has shape (number of samples, target dimension)
        """
        return