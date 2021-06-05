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


SIGMOID = Activation(lambda x: 1 / (1 + exp(-x)),
                     lambda x: x * (1 - x))
