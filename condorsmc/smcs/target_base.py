class TargetBase():
    def __init__(self, data={}):
        self.dim = 0
        self.data = data
    
    def logprior(self, x):
        raise NotImplementedError

    def loglikelihood(self, x):
        raise NotImplementedError

    def logpdf(self, x, phi=1.0):
        # Of form self.logprior(x) + phi * self.loglikelihood(x)
        raise NotImplementedError

    def logpdfgrad(self, x, phi=1.0):
        raise NotImplementedError
