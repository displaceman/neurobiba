from numpy import random, array, dot, append
from pickle import (dump, load)
from neurobiba.activations import *
from neurobiba.helpers import default_counter
import os
from abc import ABC, abstractmethod


_WEIGHTS_NAME_PREFIX = "weights_"


class BiasStrategy(ABC):
    # @abstractmethod
    def is_bias():...
    

    # @abstractmethod
    @staticmethod
    def feed_backward(__weights__, __input_layer__):...
    

    # @abstractmethod
    @staticmethod
    def feed_forward(__network__, __input_layer__):...
    
    
    # @abstractmethod
    @staticmethod
    def train(network, input_layer, correct_output, alpha):...


    @staticmethod
    def correct_outer_layer(network, correct_output, layers, alpha):
        # Корректирует крайний слой весов
        error = correct_output - layers[-1]
        delta = error * network.activation.deriv(layers[-1]) * alpha
        network.weights[-1] += layers[-2].T.dot(delta)


class WithBiasStrategy(BiasStrategy):
    @staticmethod
    def is_bias():
        return True


    @staticmethod
    def feed_backward(__weights__, __input_layer__):
        raise NotImplementedError("feed_backward работает только с весами без биаса")
    

    @staticmethod
    def feed_forward(network, input_layer):
        layers = [array([input_layer])]
        # network.__bias_strategy.__feed_forward_cycle(network, layers)
        WithBiasStrategy.__feed_forward_cycle(network, layers)
        return layers
    

    @staticmethod
    def __feed_forward_cycle(network, layers):
        for i in range(len(network.weights)):
            layers[-1] = array([append(layers[-1], 1)])
            layers.append(network.activation.fn(
                dot(layers[-1], network.weights[i])))
    

    @staticmethod
    def train(network, input_layer, correct_output, alpha):
        # Прогон через нейрость
        print('Train')
        layers = network.__bias_strategy.feed_forward(network, input_layer)
        
        # Корректирует крайний слой весов
        WithBiasStrategy.correct_outer_layer(network, correct_output, layers, alpha)

        # Корректрирует остальные слои весов
        for i in range(len(network.weights)-1):
            error = delta.dot(network.weights[len(network.weights)-1-i].T)
            delta = error * network.activation.deriv(layers[len(network.weights)-1-i]) * alpha
            delta = array([delta[0][:-1]])
            network.weights[-2-i] += layers[-3-i].T.dot(delta)


class WithoutBiasStrategy(BiasStrategy):
    @staticmethod
    def is_bias():
        return False


    @staticmethod
    def feed_backward(weights, input_layer):
        weightsr = list(reversed(weights.weights))
        for ind, i in enumerate(weightsr):
            weightsr[ind] = weightsr[ind].T

        layers = [array([input_layer])]
        len_weights = len(weightsr)

        for i in range(len_weights):
            layers.append(weights.activation.fn(dot(layers[-1], weightsr[i])))
        return layers[-1][0]
    

    @staticmethod
    def feed_forward(network, input_layer):
        print('ff no_bias')
        layers = [array([input_layer])]
        # network.__bias_strategy.__feed_forward_cycle(network, layers)
        WithoutBiasStrategy.__feed_forward_cycle(network, layers)
        return layers
    

    @staticmethod
    def __feed_forward_cycle(network, layers):
        for i in range(len(network.weights)):
            layers.append(network.activation.fn(
                dot(layers[-1], network.weights[i])))
    

    @staticmethod
    def train(network, input_layer, correct_output, alpha):
        # Прогон через нейрость
        layers = network._Weights__bias_strategy.feed_forward(network, input_layer)
        
        # Корректирует крайний слой весов
        WithoutBiasStrategy.correct_outer_layer(network, correct_output, layers, alpha)

        # Корректрирует остальные слои весов
        for i in range(len(network.weights)-1):
            error = delta.dot(network.weights[len(network.weights)-1-i].T)
            delta = error * network.activation.deriv(layers[len(network.weights)-1-i]) * alpha
            network.weights[-2-i] += layers[-3-i].T.dot(delta)


BIAS = WithBiasStrategy()

NO_BIAS = WithoutBiasStrategy()


class Weights():
    def __init__(self,
                 size = [1, 1],
                 bias = NO_BIAS,
                 name = None,
                 activation=SIGMOID):
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

        # self.bias = bias
        self.name = name if name else _WEIGHTS_NAME_PREFIX + \
            str(default_counter())
        self.activation = activation
        # if bias:
        #     self.__feed_backward_strategy = _feed_backward_with_bias
        # else:
        #     self.__feed_backward_strategy = _feed_backward_without_bias

        self.__bias_strategy = bias

        self.weights = [2*random.random((size[i]+int(bias.is_bias()), size[i+1])) - 1
                        for i in range(len(size)-1)]


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

        # # Прогон через нейрость
        # layers = self.__bias_strategy.feed_forward(self, input_layer)
        
        # # Корректирует крайний слой весов
        # error = correct_output - layers[-1]
        # delta = error * self.activation.deriv(layers[-1]) * alpha
        # self.weights[-1] += layers[-2].T.dot(delta)

        # # Корректрирует остальные слои весов
        # for i in range(len(self.weights)-1):
        #     error = delta.dot(self.weights[len(self.weights)-1-i].T)
        #     delta = error * self.activation.deriv(layers[len(self.weights)-1-i]) * alpha
        #     if self.bias: delta = array([delta[0][:-1]])
        #     self.weights[-2-i] += layers[-3-i].T.dot(delta)

        self.__bias_strategy.train(self, input_layer, correct_output, alpha)
 

    def feed_forward(self, input_layer):
        """
        Метод для получения ответа нейросети.
        Возвращает список выходных нейронов.\n
        Пример:\n
        `result = weights.feed_forward(input_layer)`\n
        `input_layer` - список входных нейронов.
        """
        # return self.__feed_forward(input_layer)[-1][0]
        return self.__bias_strategy.feed_forward(self, input_layer)[-1][0]
    

    def feed_backward(self, input_layer):
        """
        НЕ РАБОТАЕТ НА ВЕСАХ С БИАСОМ\n
        Эта функция работает так же, как `feed_forward`,
        но проводит сигнал через нейросеть в обратном направлении.\n
        В нее в качестве входного слоя подают то, что ранее считалось выходным слоем.\n
        Пример:\n
        `r = weights.feed_backward(input_layer)`\n
        """
        return self.__bias_strategy.feed_backward(self, input_layer)


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
