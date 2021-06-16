from numpy import exp, maximum, array

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


def _lrelu(x):
    return maximum(x, 0.01*x)
def _lrelu_derivative(x):
    a = []
    for i in x[0]:
        if i>=0:
            a.append(1)
        else:
            a.append(0.01)
    r = array([a])
    return r
LRELU = Activation(_lrelu, _lrelu_derivative)


def _relu(x):
    return maximum(x, 0)
def _relu_derivative(x):
    a = []
    for i in x[0]:
        if i>=0:
            a.append(1)
        else:
            a.append(0)
    r = array([a])
    return r
RELU = Activation(_relu, _relu_derivative)


def _linear(x):
    return x
def _linear_derivative(x):
    a = []
    for i in x[0]:
            a.append(1)
    r = array([a])
    return r
LINEAR = Activation(_linear, _linear_derivative)