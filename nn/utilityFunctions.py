from scipy.special import expit
import numpy as np

class sigmoid():
    @staticmethod
    def f(x):
        return expit(x)

    @staticmethod
    def df(x):
        return sigmoid.f(x) * (1 - sigmoid.f(x))


class crossEntropy():
    @staticmethod
    def f(y, a):
        """
            y -> expected
            a -> actual
        """
        assert a.shape == y.shape
        error = -np.sum(y*np.log(a) + (1 - y)*np.log(1 - a), axis = 1)
        assert error.shape[0] == a.shape[0]
        return np.average(error, axis = 0)

    @staticmethod
    def df(y, a):
        assert a.shape == y.shape
        return (a - y) / (a * (1-a) * a.shape[0])


class sqrcost():
    @staticmethod
    def f(y, a):
        """
            y -> expected
            a -> actual
        """
        assert a.shape == y.shape
        # averaging over all test cases
        return 0.5 * np.sum(np.average(np.square(y - a), axis = 0))

    @staticmethod
    def df(y, a):
        assert a.shape == y.shape
        return (a - y) / a.shape[0] # number or test cases to average them out
