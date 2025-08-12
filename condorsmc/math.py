import autograd.numpy as np  # type: ignore


def logsumexp(a, b):
    """
    Calculates the log of the sum of the exponentials of two numbers

    Args:
        a: A number
        b: A number

    Returns:
        The log of the sum of the exponentials of a and b
    """

    if a > b:
        return a + np.log(1 + np.exp(b - a))
    else:
        return b + np.log(1 + np.exp(a - b))


def logsumexpseq(a):
    """
    Calculates the log of the sum of the exponentials of a sequence of numbers

    Args:
        a: A sequence of numbers

    Returns:
        The log of the sum of the exponentials of a
    """

    return np.max(a) + np.log(np.sum(np.exp(a - np.max(a))))
