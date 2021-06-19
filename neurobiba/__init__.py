from numpy import (exp, random, array, dot, append, delete)
import random as rand
from pickle import (dump, load)
from neurobiba.activations import *
from neurobiba.helpers import default_counter
import os

_WEIGHTS_NAME_PREFIX = "weights_"


class Weights():
    def __init__(self, size=[1, 1], bias=True, name=None, activation=SIGMOID):
        """
        `size` - список слоев с количеством их нейронов.\n
        `bias` - этот флаг добавляет к каждому слою нейрон смещения.\n
        `name` - имя файла в который будет сохранять функция save_weights.\n
        `activation` - объект класса Activation который содержит в себе функцию активации и ее производную.\n
        Пример:
        `weights = Weights([3, 10, 10 ,2])`\n
        Здесь три нейрона на входном слое, 
        два промежуточных слоя по 10 нейронов и 2 нейрона на выходе.
        """

        self.bias = bias
        self.name = name if name else _WEIGHTS_NAME_PREFIX + \
            str(default_counter())
        self.activation = activation
        if bias:
            self.__feed_backward_strategy = _feed_backward_with_bias
        else:
            self.__feed_backward_strategy = _feed_backward_without_bias

        self.__weights = [
            2 * random.random((size[i] + int(bias), size[i + 1])) - 1
            for i in range(len(size) - 1)
        ]

    def __getitem__(self, key):
        return self.__weights[key]

    def __call__(self, input_layer):
        return self.feed_forward(input_layer)

    def train(self, input_layer, correct_output, alpha=0.9):
        """
        Функция для обучения нейросети 
        методом обратного распространения ошибки.
        Меняет веса.
        Ее нужно вызывать в цикле столько раз сколько потребуется для обучения.\n
        Пример:
        ```
        for i in range(1000):
            weights.training(input_layer, correct_output)
        ```
        `input_layer` - список нейронов входного слоя. 
        Они должны находится в диапазоне `0` до `1`.\n
        Например:\n
        `input_layer = [1, 0, 0.7]`\n
        `correct_output` - это список правильных выходов для заданных входов.
        Они должны находится в диапазоне от `0` до `1`.\n
        Например: `correct_output = [0.5, 1]`\n
        `alpha` - коэфициент скорости обучения. 
        Его оптимальное значение меняется в зависимости от задачи.
        """
        if self.activation != SIGMOID:
            alpha = 0.005

        # Прогон через нейрость
        layers = self.__feed_forward(input_layer)

        # Корректирует крайний слой весов
        error = correct_output - layers[-1]
        delta = error * self.activation.deriv(layers[-1]) * alpha
        self.__weights[-1] += layers[-2].T.dot(delta)

        # Корректрирует остальные слои весов
        for i in range(len(self.__weights) - 1):
            error = delta.dot(self.__weights[len(self.__weights) - 1 - i].T)
            delta = error * self.activation.deriv(
                layers[len(self.__weights) - 1 - i]) * alpha
            if self.bias: delta = array([delta[0][:-1]])
            self.__weights[-2 - i] += layers[-3 - i].T.dot(delta)

    def feed_forward(self, input_layer):
        """
        Метод для получения ответа нейросети.
        Возвращает список выходных нейронов.\n
        Пример:\n
        `result = weights.feed_forward(input_layer)`\n
        `input_layer` - список входных нейронов.
        """
        return self.__feed_forward(input_layer)[-1][0]

    def __feed_forward(self, input_layer):
        """Вычисляет и возвращает все слои"""
        layers = [array([input_layer])]

        for i in range(len(self.__weights)):
            if self.bias: layers[-1] = array([append(layers[-1], 1)])
            layers.append(
                self.activation.fn(dot(layers[-1], self.__weights[i])))

        return layers

    def feed_backward(self, input_layer):
        """
        НЕ РАБОТАЕТ НА ВЕСАХ С БИАСОМ\n
        Эта функция работает так же, как `feed_forward`,
        но проводит сигнал через нейросеть в обратном направлении.\n
        В нее в качестве входного слоя подают то, что ранее считалось выходным слоем.\n
        Пример:\n
        `r = weights.feed_backward(input_layer)`\n
        """
        return self.__feed_backward_strategy(self, input_layer)

    def mutate(self, power=1, probability=0.5):
        n = []
        for i in self.__weights:
            nn = []
            for ii in i:
                nnn = []
                for iii in ii:
                    if rand.random() < probability:
                        nnn.append(iii + (rand.random() * 2 - 1) * power)
                    else:
                        nnn.append(iii)
                nn.append(nnn)
            n.append(array(nn))
        self.__weights = n


def _feed_backward_without_bias(weights, input_layer):
    weightsr = list(reversed(weights[:]))
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
    Загрузка весов из файла.\n
    Пример использования:\n
    `weights = load_weights()`\n
    В качестве аргумента `file_name` можно указать имя файла.\n
    """

    if not os.path.exists(file_name):
        raise FileNotFoundError(f'No such file or directory: {file_name}')

    with open(file_name, 'rb') as file:
        result = load(file)
        if isinstance(result, Weights):
            print('file loaded')
            return result
        else:
            raise TypeError


def save_weights(weights, file_name=None):
    """
    Сохранение весов в файл.\n
    Пример использования:\n
    `save_weights(weights)`\n
    В качестве аргумента `file_name` можно указать имя файла.\n
    """
    if file_name is None:
        file_name = weights.name

    with open(file_name, 'wb') as file:
        dump(weights, file)
        print('file saved')
