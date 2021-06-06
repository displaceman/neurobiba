from numpy import exp


class Activation():
    def __init__(self, activation, derivative):
        self._fn = activation
        self._deriv = derivative

    @property
    def fn(self):
        return self._fn

    @property
    def deriv(self):
        return self._deriv


def sigmoid(x):
    return 1 / (1 + exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


SIGMOID = Activation(sigmoid, sigmoid_derivative)
