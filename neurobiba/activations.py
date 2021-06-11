from numpy import exp
from numpy import maximum

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


def _sigmoid(x):
    return 1 / (1 + exp(-x))
def _sigmoid_derivative(x):
    return x * (1 - x)
SIGMOID = Activation(_sigmoid, _sigmoid_derivative)


def _relu(x):
    return maximum(x, 0)
def _relu_derivative(x):
    return 1-x**0
RELU = Activation(_relu, _relu_derivative)
