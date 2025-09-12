class ExecutorBase:
    def __init__(self):
        self.num_procs = 1
        self.rank = 0

    def max(self, x):
        pass

    def sum(self, x):
        pass

    def weighted_sum(self, x, w):
        pass

    def gather(self, x):
        pass

    def bcast(self, x):
        pass

    def logsumexp(self, x):
        pass

    def cumsum(self, x):
        pass
