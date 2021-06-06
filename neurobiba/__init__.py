from numpy import (exp, random, array, dot, append)
from pickle import (dump, load)
from neurobiba.activations import *
from neurobiba.helpers import default_counter


_WEIGHTS_NAME_PREFIX = "weights_"


class Weights():
    def __init__(self,
                 size=[1, 1],
                 bias=False,
                 name=None,
                 activation=SIGMOID):
        """
        size это список слоев с количеством их нейронов.
        bias этот флаг добавляет к каждому слою нейрон смещения.
        name это имя файла в который будет сохранять функция save_weights.
        activation это объект класса Activation который содержит в себе функцию активации и ее производную

        Пример:
        `weights = Weights([3, 10, 10 ,2])`

        Здесь три нейрона на входном слое, 
        два промежуточных слоя по 10 нейронов и 2 нейрона на выходе.
        """

        self.bias = bias
        self.name = name if name else _WEIGHTS_NAME_PREFIX + \
            str(default_counter())
        self.activation = activation
        if bias:
            self._init_weights_with_bias()
        else:
            self._init_weights_with_bias()
        self.weights = [
            2*random.random((size[i]+int(bias), size[i+1])) - 1 for i in range(len(size)-1)]

    def _init_weights_with_bias(self):
        self._feed_backward_strategy = _feed_backward_with_bias
        self._feed_forward_strategy = _feed_forward_with_bias

    def _init_weights_without_bias(self):
        self._feed_backward_strategy = _feed_backward_without_bias
        self._feed_forward_strategy = _feed_forward_without_bias

    def train(self, input_layer, correct_output, alpha=0.9):
        """
        Функция для обучения нейросети 
        методом обратного распространения ошибки.
        Меняет веса.
        Ее нужно вызывать в цикле столько раз сколько потребуется для обучения.

        Пример:
        ```
        for i in range(1000):
            weights.training(input_layer, correct_output)
        ```

        `input_layer` - это список нейронов входного слоя. 
        Они должны находится в диапазоне от `0` до `1`.
        Например: `input_layer = [1, 0, 0.7]`

        `correct_output` - это список правильных выходов для заданных входов.
        Они должны находится в диапазоне от `0` до `1`.
        Например: `correct_output = [0.5, 1]`

        `alpha` - это коэфициент скорости обучения. 
        Его оптимальное значение меняется в зависимости от задачи.
        """

        layers = [array([input_layer])]
        len_weights = len(self.weights)

        for i in range(len_weights):
            if self.bias:
                layers[-1] = array([append(layers[-1], 1)])
            layers.append(self.activation.fn(dot(layers[-1], self.weights[i])))

        layers_error = []
        layers_delta = []

        layers_error.append(correct_output - layers[-1])
        layers_delta.append(
            layers_error[-1] * self.activation.deriv(layers[-1]) * alpha)

        for i in range(len_weights-1):
            layers_error.append(layers_delta[i].dot(
                self.weights[len_weights-1-i].T))
            if self.bias:
                layers_delta.append(
                    array([(layers_error[-1] * self.activation.deriv(layers[len_weights-1-i]) * alpha)[0][:-1]]))
            else:
                layers_delta.append(
                    layers_error[-1] * self.activation.deriv(layers[len_weights-1-i]) * alpha)

        for ind in range(len_weights):
            self.weights[ind] += layers[ind].T.dot(layers_delta[-1-ind])

    def feed_forward(self, input_layer):
        """
        Функция для получения ответа нейросети.
        Возвращает список выходных нейронов.

        Пример:
        `result = weights.feed_forward(input_layer)`

        `input_layer` - это список входных нейронов.
        """
        return self._feed_forward_strategy(self, input_layer)

    def feed_backward(self, input_layer):
        """
        НЕ РАБОТАЕТ НА ВЕСАХ С БИАСОМ

        Эта функция работает так же, как `feed_forward`,
        но проводит сигнал через нейросеть в обратном направлении.
        В нее в качестве входного слоя подают то, что ранее считалось выходным слоем.

        Пример:
        `r = weights.feed_backward(input_layer)`
        """
        return self._feed_backward_strategy(self, input_layer)


def _feed_forward_without_bias(weights, input_layer):
    layers = [array([input_layer])]
    len_weights = len(weights.weights)

    for i in range(len_weights):
        layers.append(weights.activation.fn(
            dot(layers[-1], weights.weights[i])))

    return layers[-1][0]


def _feed_forward_with_bias(weights, input_layer):
    layers = [array([input_layer])]
    len_weights = len(weights.weights)

    for i in range(len_weights):
        layers[-1] = array([append(layers[-1], 1)])
        layers.append(weights.activation.fn(
            dot(layers[-1], weights.weights[i])))

    return layers[-1][0]


def _feed_backward_without_bias(weights, input_layer):
    weightsr = list(reversed(weights.weights))
    for ind, i in enumerate(weightsr):
        weightsr[ind] = weightsr[ind].T

    layers = [array([input_layer])]
    len_weights = len(weightsr)

    for i in range(len_weights):
        layers.append(weights.activation.fn(dot(layers[-1], weightsr[i])))
    return layers[-1][0]


def _feed_backward_with_bias(_weights, _input_layer):
    raise NotImplementedError(
        "feed_backward работает только с весами без биаса")


def load_weights(file_name=_WEIGHTS_NAME_PREFIX + "0") -> Weights:
    """
    Загрузка весов из файла.

    Пример использования:
    `weights = load_weights()`

    В качестве аргумента `file_name` можно указать имя файла
    """

    try:
        with open(file_name, 'rb') as file:
            print('file loaded')
            return load(file)
    except FileNotFoundError:
        print(f'FileNotFoundError: No such file or directory: {file_name}')


def save_weights(weights, file_name=None):
    """
    Схранение весов в файл.

    Пример использования:
    `save_weights(weights)`

    В качестве аргумента `file_name` можно указать имя файла
    """
    if file_name is None:
        file_name = weights.name

    with open(file_name, 'wb') as file:
        dump(weights, file)
        print('file saved')
